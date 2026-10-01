"""NER: dados, treino do BERTimbau no LeNER-Br e predição."""
import random

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
    Trainer, TrainingArguments, set_seed,
)

SEED = 42
CHECKPOINT = "neuralmind/bert-base-portuguese-cased"


def fixar_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)


def carregar_dados():
    """Retorna (dataset, label_names)."""
    dataset = load_dataset("peluz/lener_br", revision="refs/pr/5")
    label_names = dataset["train"].features["ner_tags"].feature.names
    return dataset, label_names


def alinhar_labels_com_tokens(labels_originais, word_ids):
    labels_alinhadas = []
    id_palavra_anterior = None

    for word_id in word_ids:
        if word_id is None:
            # token especial ([CLS], [SEP], padding)
            labels_alinhadas.append(-100)
        elif word_id != id_palavra_anterior:
            # primeiro subtoken desta palavra
            labels_alinhadas.append(labels_originais[word_id])
        else:
            # subtoken extra da mesma palavra
            labels_alinhadas.append(-100)
        id_palavra_anterior = word_id

    return labels_alinhadas


def tokenizar_dataset(dataset, tokenizer):
    def tokenizar_e_alinhar(exemplos):
        tokenized = tokenizer(
            exemplos["tokens"],
            truncation=True,
            max_length=512,
            is_split_into_words=True,
        )
        tokenized["labels"] = [
            alinhar_labels_com_tokens(labels, tokenized.word_ids(batch_index=i))
            for i, labels in enumerate(exemplos["ner_tags"])
        ]
        return tokenized

    return dataset.map(tokenizar_e_alinhar, batched=True)


def decodificar(logits, labels, label_names):
    """Retorna (true_labels, true_predictions): tags string por sentença, sem -100."""
    predictions = np.argmax(logits, axis=-1)
    true_labels = [
        [label_names[l] for l in label if l != -100]
        for label in labels
    ]
    true_predictions = [
        [label_names[p] for p, l in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions


def treinar_ner(dataset_tokenizado, tokenizer, label_names, output_dir="./bertimbau-lener-br"):
    """Fine-tuning do BERTimbau. Retorna o Trainer já treinado."""
    seqeval = evaluate.load("seqeval")

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        true_labels, true_predictions = decodificar(logits, labels, label_names)
        r = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": r["overall_precision"],
            "recall": r["overall_recall"],
            "f1": r["overall_f1"],
            "accuracy": r["overall_accuracy"],
        }

    model = AutoModelForTokenClassification.from_pretrained(
        CHECKPOINT,
        num_labels=len(label_names),
        id2label={i: n for i, n in enumerate(label_names)},
        label2id={n: i for i, n in enumerate(label_names)},
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
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset_tokenizado["train"],
        eval_dataset=dataset_tokenizado["validation"],
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer


def prever(trainer, dataset_tokenizado, label_names, split="test", imprimir=True):
    """Prediz em `split` e devolve um dict com tudo que o resto do pipeline usa:
    logits, labels, true_labels, true_predictions, seqeval (dict completo)."""
    out = trainer.predict(dataset_tokenizado[split])
    logits, labels = out.predictions, out.label_ids
    true_labels, true_predictions = decodificar(logits, labels, label_names)

    seqeval = evaluate.load("seqeval").compute(predictions=true_predictions, references=true_labels)

    if imprimir:
        for entidade, m in seqeval.items():
            if entidade.startswith("overall"):
                continue
            print(f"{entidade:20} precision={m['precision']:.3f}  recall={m['recall']:.3f}  f1={m['f1']:.3f}  n={m['number']}")
        print("\nOverall F1:", seqeval["overall_f1"])

    return {
        "logits": logits, "labels": labels,
        "true_labels": true_labels, "true_predictions": true_predictions,
        "seqeval": seqeval,
    }
