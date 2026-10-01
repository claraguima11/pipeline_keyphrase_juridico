"""Entidades BIO, mismatches (Type-1 a 5), F1 (4 variantes) e métricas de ranking."""
from collections import Counter

import numpy as np
from scipy.special import softmax

TIPOS = ["type-1", "type-2", "type-3", "type-4", "type-5"]


# ======================= entidades e mismatches =======================

def bio_para_entidades(tokens_labels):
    """Converte lista de tags BIO em lista de entidades no formato:
    [{'label': 'PESSOA', 'start': 0, 'end': 2}, ...]"""
    entidades = []
    entidade_atual = None

    for i, label in enumerate(tokens_labels):
        if label.startswith("B-"):
            if entidade_atual:
                entidades.append(entidade_atual)
            entidade_atual = {"label": label[2:], "start": i, "end": i + 1}
        elif label.startswith("I-") and entidade_atual:
            entidade_atual["end"] = i + 1
        else:
            if entidade_atual:
                entidades.append(entidade_atual)
            entidade_atual = None

    if entidade_atual:
        entidades.append(entidade_atual)
    return entidades


def spans_se_sobrepoem(predicao, gold):
    return predicao["end"] > gold["start"] and gold["end"] > predicao["start"]


def texto_da_entidade(ent, tokens):
    return " ".join(tokens[ent["start"]:ent["end"]])


def classificar_mismatches(entidades_gold, entidades_pred, tokens):
    '''
    Classifica cada par gold/predição em Type-1 a Type-5.
    Retorna uma lista de dicts com todos os detalhes (tipo, textos, labels).
    '''
    detalhes = []
    gold_usados = set()
    pred_usadas = set()

    for i_pred, pred in enumerate(entidades_pred):
        for i_gold, gold in enumerate(entidades_gold):
            if i_gold in gold_usados:
                continue
            if spans_se_sobrepoem(pred, gold):
                gold_usados.add(i_gold)
                pred_usadas.add(i_pred)

                if pred["end"] == gold["end"] and pred["start"] == gold["start"]:
                    tipo = "type-3" if pred["label"] != gold["label"] else None
                else:
                    tipo = "type-5" if pred["label"] == gold["label"] else "type-4"

                if tipo:
                    detalhes.append({
                        "tipo": tipo,
                        "gold_texto": texto_da_entidade(gold, tokens), "gold_label": gold["label"],
                        "pred_texto": texto_da_entidade(pred, tokens), "pred_label": pred["label"],
                        "frase": " ".join(tokens),
                    })
                break

    for i_gold, gold in enumerate(entidades_gold):
        if i_gold not in gold_usados:
            detalhes.append({
                "tipo": "type-2", "gold_texto": texto_da_entidade(gold, tokens), "gold_label": gold["label"],
                "pred_texto": None, "pred_label": None, "frase": " ".join(tokens),
            })

    for i_pred, pred in enumerate(entidades_pred):
        if i_pred not in pred_usadas:
            detalhes.append({
                "tipo": "type-1", "gold_texto": None, "gold_label": None,
                "pred_texto": texto_da_entidade(pred, tokens), "pred_label": pred["label"],
                "frase": " ".join(tokens),
            })

    return detalhes


def classificar_corpus(tokens_por_frase, true_labels, true_predictions):
    """Roda `classificar_mismatches` em todas as frases. Retorna a lista de mismatches do corpus."""
    todos_detalhes = []
    for tokens_sent, labels_sent, preds_sent in zip(tokens_por_frase, true_labels, true_predictions):
        todos_detalhes.extend(classificar_mismatches(
            bio_para_entidades(labels_sent), bio_para_entidades(preds_sent), tokens_sent))
    print(Counter(d["tipo"] for d in todos_detalhes))
    return todos_detalhes


# ============================ F1 ============================

def f1_por_contagens(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def calcular_f1s(todos_detalhes, true_labels, true_predictions, resultados_juiz=None, imprimir=True):
    """Calcula as variantes de F1 de uma vez.

    estrito         : só acerto exato
    relaxado        : todo Type-5 conta como acerto
    learning_based  : só Type-5 aceitos pelo juiz deixam de ser penalizados (paper)
    estendido       : Types 3, 4 e 5 aceitos pelo juiz deixam de ser penalizados
    (learning_based e estendido só são calculados se `resultados_juiz` for passado)
    """
    q = Counter(d["tipo"] for d in todos_detalhes)
    t1, t2, t3, t4, t5 = (q[t] for t in TIPOS)

    total_gold = sum(len(bio_para_entidades(l)) for l in true_labels)
    total_pred = sum(len(bio_para_entidades(p)) for p in true_predictions)

    erros_gold = t2 + t3 + t4 + t5
    erros_pred = t1 + t3 + t4 + t5
    tp_exato = total_gold - erros_gold
    assert tp_exato + erros_gold == total_gold
    assert tp_exato + erros_pred == total_pred

    f1s = {
        "estrito": f1_por_contagens(tp_exato, erros_pred, erros_gold),
        "relaxado": f1_por_contagens(tp_exato + t5, t1 + t3 + t4, t2 + t3 + t4),
    }

    if resultados_juiz is not None:
        aceitos = Counter(r["tipo"] for r in resultados_juiz if r["decisao"] == "aceito")
        a1, a3, a4, a5 = aceitos["type-1"], aceitos["type-3"], aceitos["type-4"], aceitos["type-5"]
        r3, r4, r5 = t3 - a3, t4 - a4, t5 - a5

        f1s["learning_based"] = f1_por_contagens(tp_exato + a5, t1 + t3 + t4 + r5, t2 + t3 + t4 + r5)
        f1s["estendido"] = f1_por_contagens(
            tp_exato + a3 + a4 + a5, t1 + r3 + r4 + r5, t2 + r3 + r4 + r5)

        if imprimir:
            print(f"Type-3 aceitos pelo judge: {a3}/{t3}")
            print(f"Type-4 aceitos pelo judge: {a4}/{t4}")
            print(f"Type-5 aceitos pelo judge: {a5}/{t5}")
            print(f"[fora do F1, sem gold correspondente] Type-1 aceitos pelo judge: {a1}/{t1}")

    if imprimir:
        for nome, m in f1s.items():
            print(f"F1 {nome:15} {m['f1']:.3f}  (P={m['precision']:.3f}  R={m['recall']:.3f})")
    return f1s


# ============================ ranking ============================

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


def avaliar_ranking(tokens_por_frase, pred, agregacao="media", imprimir=True):
    """R-precision, MRR e Bpref no corpus, ranqueando as entidades previstas pela
    confiança do NER. `pred` é o dict retornado por `ner.prever`."""
    confiancas = confiancas_por_token(pred["logits"], pred["labels"])

    lista_ranked_candidatos, lista_gold = [], []
    for tokens_sent, labels_sent, preds_sent, conf_sent in zip(
            tokens_por_frase, pred["true_labels"], pred["true_predictions"], confiancas):
        ranking = montar_ranking(bio_para_entidades(preds_sent), conf_sent, tokens_sent, agregacao)
        gold_textos = {texto_da_entidade(ent, tokens_sent) for ent in bio_para_entidades(labels_sent)}
        lista_ranked_candidatos.append(ranking)
        lista_gold.append(gold_textos)

    resultado = avaliar_corpus(lista_ranked_candidatos, lista_gold)
    if imprimir:
        print(resultado)
    return resultado


# ============================ exemplos ============================

def mais_curto(lista, limite=160):
    if not lista:
        return None
    curtos = [d for d in lista if len(d["frase"]) <= limite]
    pool = curtos if curtos else lista
    return min(pool, key=lambda d: len(d["frase"]))


def mostrar_exemplos(todos_detalhes, resultados_juiz=None):
    """Imprime o exemplo mais curto de cada tipo de mismatch (e, se houver juiz,
    de cada decisão aceito/rejeitado por tipo)."""
    print("EXEMPLO REAL DE CADA TIPO DE MISMATCH")
    for tipo in TIPOS:
        exemplo = mais_curto([d for d in todos_detalhes if d["tipo"] == tipo])
        print(f"\n--- {tipo.upper()} ---")
        if exemplo is None:
            print("(nenhum caso no conjunto de teste)")
            continue
        print(f"Frase: {exemplo['frase']}")
        print(f"Gold : {exemplo['gold_texto']!r}  ({exemplo['gold_label']})")
        print(f"Pred : {exemplo['pred_texto']!r}  ({exemplo['pred_label']})")

    if resultados_juiz is None:
        return
    print("\nAUTOMATIC JUDGE — exemplos por tipo")
    for tipo in TIPOS:
        rs = [r for r in resultados_juiz if r["tipo"] == tipo]
        print(f"\n--- {tipo.upper()} — {dict(Counter(r['decisao'] for r in rs))} ---")
        for decisao in ["aceito", "rejeitado"]:
            exemplo = mais_curto([r for r in rs if r["decisao"] == decisao])
            if exemplo is None:
                print(f"\n[{decisao}] (nenhum caso)")
                continue
            print(f"\n[{decisao}]")
            print(f"Frase: {exemplo['frase']}")
            print(f"Gold : {exemplo['gold_texto']!r}  ({exemplo['gold_label']})")
            print(f"Pred : {exemplo['pred_texto']!r}  ({exemplo['pred_label']})")
            print(f"Juiz classificou o texto previsto como: {exemplo['label_juiz']}")
