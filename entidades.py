"""Entidades BIO e classificação de mismatches (Type-1 a Type-5)."""


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
