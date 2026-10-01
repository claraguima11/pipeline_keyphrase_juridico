"""Automatic Judge: candidatos 'other' (noun chunks do spaCy) e avaliação dos erros."""
import re

import torch
import torch.nn.functional as F

from entidades import bio_para_entidades, texto_da_entidade


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
    """
    Retorna início e fim, em caracteres, de cada token
    na frase construída com ' '.join(tokens).
    """
    offsets = []
    posicao = 0

    for token in tokens:
        inicio = posicao
        fim = inicio + len(token)
        offsets.append((inicio, fim))
        posicao = fim + 1  # espaço inserido pelo join

    return offsets


def candidatos_other_da_frase(exemplo, label_names, nlp):
    tokens = exemplo["tokens"]
    tags_strings = [label_names[tag] for tag in exemplo["ner_tags"]]
    entidades_gold = bio_para_entidades(tags_strings)

    frase = " ".join(tokens)
    offsets = offsets_dos_tokens(tokens)

    # Cada entidade gold passa de índices de tokens para posições na frase.
    spans_gold = [
        (offsets[ent["start"]][0], offsets[ent["end"] - 1][1])
        for ent in entidades_gold
    ]

    doc = nlp(frase)
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
        "gold": [
            (texto_da_entidade(ent, tokens), ent["label"])
            for ent in entidades_gold
        ],
        "candidatos_other": candidatos,
    }


def classificar_span(texto, modelo_aux, tokenizer_aux, id2label_aux):
    """Retorna a classe e a confiança do automatic judge para um span."""
    inputs = tokenizer_aux(texto, return_tensors="pt", truncation=True)
    inputs = {chave: valor.to(modelo_aux.device) for chave, valor in inputs.items()}

    modelo_aux.eval()
    with torch.no_grad():
        probabilidades = F.softmax(modelo_aux(**inputs).logits, dim=-1)[0]

    classe_id = probabilidades.argmax().item()
    return id2label_aux[classe_id], probabilidades[classe_id].item()


def avaliar_erro_com_juiz(erro, modelo_aux, tokenizer_aux, id2label_aux):
    """Avalia um mismatch; Type-2 não tem span previsto e não é avaliável."""
    if erro["tipo"] == "type-2":
        return {
            **erro, "label_juiz": None, "confianca_juiz": None,
            "decisao": "nao_avaliavel",
            "motivo": "Não há span previsto para o judge classificar.",
        }

    label_juiz, confianca = classificar_span(erro["pred_texto"], modelo_aux, tokenizer_aux, id2label_aux)
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
