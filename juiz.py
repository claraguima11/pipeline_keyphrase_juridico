"""Automatic Judge (classificador auxiliar com classe 'other')."""
import random
import re
from collections import Counter

import numpy as np
import spacy
import torch
import torch.nn.functional as F
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
    Trainer, TrainingArguments,
)

from metricas import bio_para_entidades, texto_da_entidade

# CHECKPOINT = "adalbertojunior/distilbert-portuguese-cased"  # mais rápida, menos testada - mais parecida com a proposta do paper
CHECKPOINT = "neuralmind/bert-base-portuguese-cased"          # opção completa - mais pesada, já validada


def limpar_chunk(texto):
    texto = texto.strip()
    # Remove somente pontuação no começo e no fim.
    texto = re.sub(r"^[^\wÀ-ÿ]+", "", texto)
    texto = re.sub(r"[^\wÀ-ÿ]+$", "", texto)
    texto = texto.strip()
    # Descarta chunks vazios, só numéricos ou sem letras.
    if len(texto) < 3 or not any(letra.isalpha() for letra in texto):
        return None
    return texto


def offsets_dos_tokens(tokens):
    """Início e fim, em caracteres, de cada token na frase ' '.join(tokens)."""
    offsets = []
    posicao = 0
    for token in tokens:
        inicio = posicao
        fim = inicio + len(token)
        offsets.append((inicio, fim))
        posicao = fim + 1  # espaço inserido pelo join
    return offsets


class Juiz:
    """Uso:
        juiz = Juiz(dataset, label_names)
        juiz.treinar()
        resultados = juiz.avaliar_erros(todos_detalhes)
    """

    def __init__(self, dataset, label_names, spacy_model="pt_core_news_lg", checkpoint=CHECKPOINT):
        self.dataset = dataset
        self.label_names = label_names
        self.checkpoint = checkpoint
        self.nlp = spacy.load(spacy_model)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.modelo = None
        self.trainer = None

    # ---------- dataset auxiliar ----------

    def candidatos_other_da_frase(self, exemplo):
        tokens = exemplo["tokens"]
        tags_strings = [self.label_names[tag] for tag in exemplo["ner_tags"]]
        entidades_gold = bio_para_entidades(tags_strings)

        frase = " ".join(tokens)
        offsets = offsets_dos_tokens(tokens)

        # Cada entidade gold passa de índices de tokens para posições na frase.
        spans_gold = [
            (offsets[ent["start"]][0], offsets[ent["end"] - 1][1])
            for ent in entidades_gold
        ]

        doc = self.nlp(frase)
        candidatos = []
        for chunk in doc.noun_chunks:
            sobrepoe_entidade_gold = any(
                chunk.start_char < fim_gold and inicio_gold < chunk.end_char
                for inicio_gold, fim_gold in spans_gold
            )
            if not sobrepoe_entidade_gold:
                chunk_limpo = limpar_chunk(chunk.text)
                if chunk_limpo is not None:
                    candidatos.append(chunk_limpo)

        return {
            "frase": frase,
            "gold": [(texto_da_entidade(ent, tokens), ent["label"]) for ent in entidades_gold],
            "candidatos_other": candidatos,
        }

    def montar_pares(self, split):
        """Pares (texto, label): gold do split + 'other' amostrados
        (nº de 'other' = média de exemplos por classe)."""
        positivos, candidatos_other = [], []
        for exemplo in split:
            r = self.candidatos_other_da_frase(exemplo)
            positivos.extend(r["gold"])
            candidatos_other.extend(r["candidatos_other"])

        # Remove duplicatas, preservando a primeira forma encontrada.
        candidatos_other = list(dict.fromkeys(candidatos_other))

        contagem = Counter(label for _, label in positivos)
        limite_other = round(sum(contagem.values()) / len(contagem))
        other_amostrados = random.sample(candidatos_other, k=min(limite_other, len(candidatos_other)))

        pares = positivos + [(texto, "other") for texto in other_amostrados]
        random.shuffle(pares)
        print("Distribuição:", Counter(label for _, label in pares))
        return pares

    # ---------- treino ----------

    def treinar(self, output_dir="./classificador-auxiliar"):
        pares_treino = self.montar_pares(self.dataset["train"])
        pares_val = self.montar_pares(self.dataset["validation"])

        tags_unicas = sorted({label for _, label in pares_treino})
        self.id2label = {i: label for i, label in enumerate(tags_unicas)}
        label2id = {label: i for i, label in enumerate(tags_unicas)}
        print("Classes do juiz:", tags_unicas)

        def to_dataset(pares):
            ds = Dataset.from_dict({
                "text": [t for t, _ in pares],
                "label": [label2id[l] for _, l in pares],
            })
            return ds.map(lambda ex: self.tokenizer(ex["text"], truncation=True), batched=True)

        def compute_metrics_aux(eval_preds):
            logits, labels = eval_preds
            predictions = np.argmax(logits, axis=-1)
            return {
                "accuracy": accuracy_score(labels, predictions),
                "f1": f1_score(labels, predictions, average="macro"),
            }

        modelo = AutoModelForSequenceClassification.from_pretrained(
            self.checkpoint, num_labels=len(tags_unicas),
            id2label=self.id2label, label2id=label2id,
        )
        args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
        )
        self.trainer = Trainer(
            model=modelo,
            args=args,
            train_dataset=to_dataset(pares_treino),
            eval_dataset=to_dataset(pares_val),
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            processing_class=self.tokenizer,
            compute_metrics=compute_metrics_aux,
        )
        self.trainer.train()
        self.modelo = self.trainer.model
        return self.trainer

    # ---------- julgamento ----------

    def classificar_span(self, texto):
        """Retorna a classe e a confiança do automatic judge para um span."""
        inputs = self.tokenizer(texto, return_tensors="pt", truncation=True)
        inputs = {chave: valor.to(self.modelo.device) for chave, valor in inputs.items()}

        self.modelo.eval()
        with torch.no_grad():
            probabilidades = F.softmax(self.modelo(**inputs).logits, dim=-1)[0]

        classe_id = probabilidades.argmax().item()
        return self.id2label[classe_id], probabilidades[classe_id].item()

    def avaliar_erro(self, erro):
        """Avalia um mismatch; Type-2 não tem span previsto e não é avaliável."""
        if erro["tipo"] == "type-2":
            return {
                **erro, "label_juiz": None, "confianca_juiz": None,
                "decisao": "nao_avaliavel",
                "motivo": "Não há span previsto para o judge classificar.",
            }

        label_juiz, confianca = self.classificar_span(erro["pred_texto"])
        if label_juiz == erro["pred_label"]:
            decisao = "aceito"
            motivo = "Judge confirmou a classe prevista pelo NER."
        elif label_juiz == erro["gold_label"]:
            decisao = "rejeitado"
            motivo = "Judge confirmou a classe gold, não a prevista pelo NER."
        else:
            decisao = "rejeitado"
            motivo = "Judge não confirmou a classe prevista pelo NER."

        return {
            **erro, "label_juiz": label_juiz, "confianca_juiz": confianca,
            "decisao": decisao, "motivo": motivo,
        }

    def avaliar_erros(self, erros, imprimir=True):
        """Avalia todos os mismatches e imprime o resumo por tipo."""
        resultados = [self.avaliar_erro(e) for e in erros]
        if imprimir:
            for tipo in ["type-1", "type-2", "type-3", "type-4", "type-5"]:
                print(f"{tipo}: {dict(Counter(r['decisao'] for r in resultados if r['tipo'] == tipo))}")
        return resultados
