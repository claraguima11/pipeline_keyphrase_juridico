"""Métricas: F1 por contagens e métricas de ranking (R-precision, MRR, Bpref)."""
import numpy as np
from scipy.special import softmax

from entidades import texto_da_entidade


def f1_por_contagens(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def r_precision(ranked_candidatos, gold):
    if len(gold) == 0:
        return None
    n = len(gold)
    top_n = ranked_candidatos[:n]
    acertos = sum(1 for candidato in top_n if candidato in gold)
    return acertos / n


def reciprocal_rank(ranked_candidatos, gold):
    if len(gold) == 0:
        return None
    for i, candidato in enumerate(ranked_candidatos):
        if candidato in gold:
            return 1 / (i + 1)
    return 0.0


def bpref(ranked_candidatos, gold):
    if len(gold) == 0:
        return None
    r = len(gold)
    n = 0
    nr = 0
    soma = 0
    for candidato in ranked_candidatos:
        if candidato not in gold:
            n += 1
    for i, candidato in enumerate(ranked_candidatos):
        if candidato in gold:
            if min(r, n) == 0:
                soma += 1
            else:
                soma += 1 - (min(nr, r) / min(r, n))
        else:
            nr += 1
    return soma / r


def avaliar_corpus(lista_ranked_candidatos, lista_gold):
    rprec = [r_precision(rc, g) for rc, g in zip(lista_ranked_candidatos, lista_gold)]
    rprec_validos = [r for r in rprec if r is not None]
    rr = [reciprocal_rank(rc, g) for rc, g in zip(lista_ranked_candidatos, lista_gold)]
    rr_validos = [r for r in rr if r is not None]
    bp = [bpref(rc, g) for rc, g in zip(lista_ranked_candidatos, lista_gold)]
    bp_validos = [r for r in bp if r is not None]

    return {
        "r_precision_medio": sum(rprec_validos) / len(rprec_validos),
        "mrr": sum(rr_validos) / len(rr_validos),
        "bpref_medio": sum(bp_validos) / len(bp_validos),
    }


def confianca_de_cada_token(probs_sentenca, predictions_sentenca):
    confiancas = []
    for i, token in enumerate(probs_sentenca):
        classe_prevista = predictions_sentenca[i]
        conf_token = probs_sentenca[i][classe_prevista]
        confiancas.append(conf_token)
    return confiancas


def confiancas_por_token(logits, labels):
    probs = softmax(logits, axis=-1)
    predictions = np.argmax(logits, axis=-1)

    todas_confiancas = []
    for probs_sentenca, predictions_sentenca, labels_sentenca in zip(probs, predictions, labels):
        confiancas_sentenca = confianca_de_cada_token(probs_sentenca, predictions_sentenca)
        true_confiancas = [confiancas_sentenca[i] for i, label in enumerate(labels_sentenca) if label != -100]
        todas_confiancas.append(true_confiancas)
    return todas_confiancas


def confianca_span(entidade, confiancas_sentenca, agregacao="media"):
    if agregacao == "media":
        return sum(confiancas_sentenca[entidade['start']:entidade['end']]) / (entidade['end'] - entidade['start'])
    if agregacao == "minimo":
        return min(confiancas_sentenca[entidade['start']:entidade['end']])


def montar_ranking(entidades_pred, confiancas_sentenca, tokens_sentenca, agregacao="media"):
    entidades_pred_ordenadas = sorted(entidades_pred, key=lambda ent: confianca_span(ent, confiancas_sentenca, agregacao), reverse=True)
    return [texto_da_entidade(ent, tokens_sentenca) for ent in entidades_pred_ordenadas]
