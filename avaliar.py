"""
avaliar.py — entrada simples pro pipeline de avaliação.

Recebe arquivos JSONL exportados do Doccano e devolve as métricas: tipos de erro
(Type-1 a 5), F1 (estrito, relaxado, learning-based, estendido) e ranking
(R-precision, MRR, Bpref).

Dois jeitos de usar (Python / Colab):

    from avaliar import avaliar

    # 1) comparar dois arquivos: gold (referência) x predição. Não treina nada.
    r = avaliar("gold.jsonl", "pred.jsonl")                  # F1 estrito e relaxado
    r = avaliar("gold.jsonl", "pred.jsonl", usar_juiz=True)  # + learning-based e estendido

    # 2) um arquivo só: treina o modelo nos documentos e avalia no teste (precisa de GPU)
    r = avaliar("gold.jsonl", labels="labels.json")

Terminal:  python avaliar.py gold.jsonl pred.jsonl [--labels labels.json] [--juiz]

Formato do JSONL (uma linha por documento, offsets de caractere):
    {"text": "...", "label": [[inicio, fim, "Tipo"], ...], "meta": {"id_doc": "..."}}
    - também aceita "entities": [{"start_offset", "end_offset", "label"}]
    - no arquivo de predição, "text" pode faltar (usa o do gold) e a entidade pode ter score:
      [inicio, fim, "Tipo", 0.93] -> habilita R-precision / MRR / Bpref
    - os documentos dos dois arquivos são pareados por meta.id_doc (ou pela ordem das linhas)

Arquivo de labels (opcional; sem ele os labels saem do próprio JSONL):
    .json -> export de labels do Doccano (lista de {"text": "Tipo", ...}) ou lista de strings
    .txt  -> um label por linha

Antes de comparar, o texto vira palavras + tags BIO e é dividido em trechos de até
`max_palavras` palavras. BIO só aceita uma tag por palavra: quando duas entidades de um mesmo
arquivo se sobrepõem, fica a mais longa (a quantidade descartada é impressa).
"""
import json
import random
import re
from collections import Counter

# Troque pelos equivalentes do idioma/domínio dos seus documentos (ex.: "bert-base-cased" e
# "en_core_web_lg" pra inglês). O spaCy só é usado pelo juiz.
MODELO_PADRAO = "neuralmind/bert-base-portuguese-cased"
SPACY_PADRAO = "pt_core_news_lg"

PALAVRA = re.compile(r"\w+|[^\w\s]", re.UNICODE)


# ============================ leitura ============================

def _limpar_label(nome):
    # '-' separa B-/I- do tipo nas tags BIO, então não pode aparecer dentro do nome
    return str(nome).strip().replace("-", "_")


def ler_labels(caminho):
    """Lê o arquivo de labels (.json do Doccano, .json com lista de strings, ou .txt)."""
    if str(caminho).lower().endswith(".json"):
        with open(caminho, encoding="utf-8") as f:
            dados = json.load(f)
        if isinstance(dados, dict):
            dados = dados.get("labels", list(dados))
        nomes = [d if isinstance(d, str) else d.get("text") or d.get("name") for d in dados]
    else:
        with open(caminho, encoding="utf-8") as f:
            nomes = [linha.strip() for linha in f if linha.strip()]
    return [_limpar_label(n) for n in nomes if n]


def ler_jsonl(caminho):
    """Retorna lista de (id, texto, [(ini, fim, label), ...], scores).
    `texto` pode ser None (arquivo só com labels); `scores` = {(ini, fim, label): score},
    preenchido quando a entidade tem um 4º elemento: [ini, fim, "Tipo", score]."""
    docs = []
    with open(caminho, encoding="utf-8") as f:
        for i, linha in enumerate(f):
            if not linha.strip():
                continue
            d = json.loads(linha)
            ents, scores = [], {}
            if "label" in d:
                brutas = [(int(e[0]), int(e[1]), _limpar_label(e[2]), e[3] if len(e) > 3 else None)
                          for e in d["label"]]
            else:
                brutas = [(int(e["start_offset"]), int(e["end_offset"]), _limpar_label(e["label"]),
                           e.get("score")) for e in d.get("entities", [])]
            for a, b, l, sc in brutas:
                ents.append((a, b, l))
                if sc is not None:
                    scores[(a, b, l)] = float(sc)
            id_doc = (d.get("meta") or {}).get("id_doc") or d.get("id") or f"doc{i}"
            docs.append((id_doc, d.get("text"), ents, scores))
    return docs


# ============================ offsets -> tokens + BIO ============================

def sem_sobreposicao(ents):
    """Mantém a entidade mais longa quando duas se sobrepõem. Retorna (mantidas, n_descartadas)."""
    mantidas = []
    for e in sorted(ents, key=lambda e: (-(e[1] - e[0]), e[0])):   # mais longas primeiro
        if all(e[1] <= m[0] or m[1] <= e[0] for m in mantidas):
            mantidas.append(e)
    return sorted(mantidas), len(ents) - len(mantidas)


def _paragrafos(texto, tokens):
    """Índices [ini, fim) dos tokens de cada parágrafo (separador: linha em branco)."""
    grupos, atual = [], [0]
    for i in range(1, len(tokens)):
        entre = texto[tokens[i - 1][2]:tokens[i][1]]
        if re.search(r"\n\s*\n", entre):
            grupos.append((atual[0], i))
            atual = [i]
    grupos.append((atual[0], len(tokens)))
    return grupos


def _trechos(texto, tokens, limite, pesos=None):
    """Divide os tokens em janelas [ini, fim) cujo peso total é <= limite. Peso de cada palavra =
    nº de subtokens do BERT (ou 1 por palavra, se `pesos` for None). Junta parágrafos curtos e
    corta parágrafos longos de preferência depois de pontuação final."""
    if pesos is None:
        pesos = [1] * len(tokens)

    def peso(a, b):
        return sum(pesos[a:b])

    pedacos = []
    for ini, fim in _paragrafos(texto, tokens):
        while peso(ini, fim) > limite:
            corte, acumulado = ini, 0
            while corte < fim and acumulado + pesos[corte] <= limite:
                acumulado += pesos[corte]
                corte += 1
            corte = max(corte, ini + 1)
            for k in range(corte, max(ini + (corte - ini) // 2, ini), -1):
                if tokens[k - 1][0] in {".", ";", ":"}:
                    corte = k
                    break
            pedacos.append((ini, corte))
            ini = corte
        if fim > ini:
            pedacos.append((ini, fim))

    janelas, atual = [], None
    for ini, fim in pedacos:
        if atual and atual[1] == ini and peso(atual[0], fim) <= limite:
            atual = (atual[0], fim)
        else:
            if atual:
                janelas.append(atual)
            atual = (ini, fim)
    if atual:
        janelas.append(atual)
    return janelas


def doc_para_exemplos(texto, entidades, limite=100, contar=None):
    """Um documento -> lista de {'tokens': [...], 'tags': ['O', 'B-X', 'I-X', ...]}.
    `limite` = máx. de palavras por trecho; com `contar` (palavras -> nº de subtokens de cada
    uma) o limite passa a ser em subtokens."""
    tokens = [(m.group(), m.start(), m.end()) for m in PALAVRA.finditer(texto)]
    pesos = contar([t[0] for t in tokens]) if contar and tokens else None
    entidades, _ = sem_sobreposicao(entidades)

    # cada token recebe o índice da entidade que o cobre (ou None)
    dono = [None] * len(tokens)
    for j, (ini, fim, _) in enumerate(entidades):
        for i, (_, t_ini, t_fim) in enumerate(tokens):
            if t_ini < fim and ini < t_fim:
                dono[i] = j

    exemplos = []
    for a, b in _trechos(texto, tokens, limite, pesos):
        tags, anterior = [], None
        for i in range(a, b):
            j = dono[i]
            if j is None:
                tags.append("O")
            else:
                tags.append(("I-" if j == anterior else "B-") + entidades[j][2])
            anterior = j
        exemplos.append({"tokens": [t[0] for t in tokens[a:b]], "tags": tags})
    return exemplos


# ============================ dataset ============================

def carregar_doccano(jsonl, labels=None, max_palavras=100, prop_val=0.2, prop_teste=0.2, seed=42,
                     tokenizer=None, max_subtokens=400):
    """Equivalente ao `ner.carregar_dados()`, mas lendo o JSONL do Doccano.
    Retorna (dataset, label_names) no mesmo formato que o resto do pipeline usa.
    Com `tokenizer`, os trechos respeitam `max_subtokens` (o BERT aceita 512); sem ele,
    `max_palavras`."""
    from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value

    docs = ler_jsonl(jsonl)
    if len(docs) < 3:
        raise ValueError("Preciso de pelo menos 3 documentos (treino, validação e teste).")

    nomes = ler_labels(labels) if labels else []
    achados = sorted({l for _, _, ents, _ in docs for _, _, l in ents})
    faltando = [l for l in achados if l not in nomes]
    if faltando and labels:
        print("Aviso: labels no JSONL que não estão no arquivo de labels (adicionados):", faltando)
    nomes += faltando
    label_names = ["O"] + [f"{p}-{l}" for l in nomes for p in ("B", "I")]
    tag2id = {t: i for i, t in enumerate(label_names)}

    limite, contar = max_palavras, None
    if tokenizer is not None:
        limite = max_subtokens
        contar = lambda palavras: [max(1, len(ids)) for ids in
                                   tokenizer(palavras, add_special_tokens=False)["input_ids"]]

    descartadas = sum(sem_sobreposicao(e)[1] for _, _, e, _ in docs)
    total = sum(len(e) for _, _, e, _ in docs)
    print(f"{len(docs)} documentos, {total} entidades "
          f"({descartadas} descartadas por sobreposição, ficou a mais longa)")

    ordem = list(range(len(docs)))
    random.Random(seed).shuffle(ordem)
    n_val = max(1, round(len(docs) * prop_val))
    n_teste = max(1, round(len(docs) * prop_teste))
    grupos = {
        "validation": ordem[:n_val],
        "test": ordem[n_val:n_val + n_teste],
        "train": ordem[n_val + n_teste:],
    }

    features = Features({
        "tokens": Sequence(Value("string")),
        "ner_tags": Sequence(ClassLabel(names=label_names)),
    })
    splits = {}
    for nome, indices in grupos.items():
        exemplos = []
        for i in indices:
            _, texto, ents, _ = docs[i]
            exemplos.extend(doc_para_exemplos(texto, ents, limite, contar))
        print(f"  {nome:10} {len(indices)} docs -> {len(exemplos)} trechos")
        splits[nome] = Dataset.from_dict({
            "tokens": [e["tokens"] for e in exemplos],
            "ner_tags": [[tag2id[t] for t in e["tags"]] for e in exemplos],
        }, features=features)
    return DatasetDict(splits), label_names


# ============================ comparar dois arquivos ============================

def _parear(docs_gold, docs_pred):
    """Pareia os documentos dos dois arquivos pelo id (meta.id_doc ou id); sem ids em comum,
    pela ordem das linhas."""
    por_id = {d[0]: d for d in docs_pred}
    if not any(g[0] in por_id for g in docs_gold):
        if len(docs_gold) != len(docs_pred):
            raise ValueError("Os ids não batem e os arquivos têm número de documentos diferente.")
        print("Aviso: ids não batem entre os arquivos, pareando pela ordem das linhas.")
        return list(zip(docs_gold, docs_pred))
    pares = []
    for g in docs_gold:
        p = por_id.get(g[0])
        if p is None:
            print(f"Aviso: {g[0]} não está no arquivo de predições (conta como sem nenhuma entidade).")
            p = (g[0], None, [], {})
        pares.append((g, p))
    extras = set(por_id) - {g[0] for g in docs_gold}
    if extras:
        print("Aviso: documentos só no arquivo de predições (ignorados):", sorted(extras))
    return pares


def _norm(texto):
    return " ".join(texto.split())


def comparar(gold, pred, labels=None, max_palavras=100, usar_juiz=False, imprimir=True,
             modelo=MODELO_PADRAO, spacy_model=SPACY_PADRAO):
    """Compara dois arquivos JSONL no formato do Doccano: `gold` (referência) e `pred`
    (o que está sendo avaliado). Não treina o NER.

    Sempre calcula: tipos de erro (Type-1 a 5) e F1 estrito e relaxado.
    Com usar_juiz=True: também treina o juiz (no gold) e calcula learning-based e estendido.
    Se as entidades de `pred` tiverem score ([ini, fim, "Tipo", score]): também R-precision,
    MRR e Bpref (por documento, ordenando as entidades pelo score)."""
    from metricas import avaliar_corpus, calcular_f1s, classificar_corpus

    pares = _parear(ler_jsonl(gold), ler_jsonl(pred))

    tokens, tags_gold, tags_pred = [], [], []
    ranking_pred, ranking_gold, descartadas = [], [], 0
    tem_score = True
    for (id_doc, texto, ents_g, _), (_, texto_p, ents_p, scores_p) in pares:
        if texto_p is not None and texto_p != texto:
            raise ValueError(f"O texto de {id_doc} é diferente nos dois arquivos; os offsets não casam.")
        descartadas += sem_sobreposicao(ents_g)[1] + sem_sobreposicao(ents_p)[1]
        for g, p in zip(doc_para_exemplos(texto, ents_g, max_palavras),
                        doc_para_exemplos(texto, ents_p, max_palavras)):
            tokens.append(g["tokens"])
            tags_gold.append(g["tags"])
            tags_pred.append(p["tags"])

        if not all(e in scores_p for e in ents_p):
            tem_score = False
        ordenadas = sorted(ents_p, key=lambda e: -scores_p.get(e, 0.0))
        vistos = list(dict.fromkeys(_norm(texto[a:b]) for a, b, _ in ordenadas))
        ranking_pred.append(vistos)
        ranking_gold.append({_norm(texto[a:b]) for a, b, _ in ents_g})

    print(f"{len(pares)} documentos comparados, {len(tokens)} trechos "
          f"({descartadas} entidades descartadas por sobreposição nos dois arquivos, ficou a mais longa)")

    detalhes = classificar_corpus(tokens, tags_gold, tags_pred)

    resultados_juiz = None
    if usar_juiz:
        from juiz import Juiz
        dataset, label_names = carregar_doccano(gold, labels, max_palavras)
        juiz = Juiz(dataset, label_names, spacy_model=spacy_model, checkpoint=modelo)
        juiz.treinar()
        resultados_juiz = juiz.avaliar_erros(detalhes)

    f1s = calcular_f1s(detalhes, tags_gold, tags_pred, resultados_juiz, imprimir=imprimir)

    ranking = None
    if tem_score and any(ranking_pred):
        ranking = avaliar_corpus(ranking_pred, ranking_gold)
        if imprimir:
            print(ranking)
    elif imprimir:
        print("Ranking (R-precision/MRR/Bpref) pulado: as entidades do arquivo de predições não têm score.")

    return {
        "f1": f1s,
        "ranking": ranking,
        "mismatches": dict(Counter(d["tipo"] for d in detalhes)),
        "detalhes": detalhes,
        "resultados_juiz": resultados_juiz,
    }


# ============================ pipeline completo ============================

def avaliar(gold, pred=None, labels=None, max_palavras=100, usar_juiz=False, agregacao="media", seed=42,
            epocas=3, max_subtokens=400, modelo=MODELO_PADRAO, spacy_model=SPACY_PADRAO,
            lr=2e-5, batch_size=16):
    """Ponto de entrada.
    - avaliar(gold, pred): compara dois JSONL do Doccano (ver `comparar`). Não treina nada.
    - avaliar(gold): treina o modelo (padrão: BERTimbau) nos documentos do `gold` e avalia no teste
      (NER -> mismatches -> juiz -> 4 F1s -> ranking). Precisa de GPU (Colab).
      Com poucos documentos o modelo precisa de mais épocas pra aprender: tente epocas=15.
    - modelo / spacy_model: troque pelos do idioma dos seus documentos; lr, batch_size, epocas
      e max_subtokens também são ajustáveis."""
    if pred is not None:
        return comparar(gold, pred, labels, max_palavras, usar_juiz, modelo=modelo, spacy_model=spacy_model)
    jsonl = gold

    import ner
    from juiz import Juiz
    from metricas import avaliar_ranking, calcular_f1s, classificar_corpus
    from transformers import AutoTokenizer

    ner.fixar_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(modelo)
    dataset, label_names = carregar_doccano(jsonl, labels, max_palavras, seed=seed,
                                            tokenizer=tokenizer, max_subtokens=max_subtokens)
    dataset_tok = ner.tokenizar_dataset(dataset, tokenizer)
    trainer = ner.treinar_ner(dataset_tok, tokenizer, label_names, epocas=epocas,
                              checkpoint=modelo, lr=lr, batch_size=batch_size)
    pred = ner.prever(trainer, dataset_tok, label_names)

    tokens_teste = dataset["test"]["tokens"]
    for i, (t, l) in enumerate(zip(tokens_teste, pred["true_labels"])):
        if len(t) != len(l):
            raise ValueError(f"Trecho {i} passou do limite do BERT e foi truncado; "
                             f"diminua max_subtokens (atual: {max_subtokens}).")

    detalhes = classificar_corpus(tokens_teste, pred["true_labels"], pred["true_predictions"])

    juiz = Juiz(dataset, label_names, spacy_model=spacy_model, checkpoint=modelo)
    juiz.treinar()
    resultados_juiz = juiz.avaliar_erros(detalhes)

    f1s = calcular_f1s(detalhes, pred["true_labels"], pred["true_predictions"], resultados_juiz)
    ranking = avaliar_ranking(tokens_teste, pred, agregacao=agregacao)

    return {
        "f1": f1s,
        "ranking": ranking,
        "seqeval_f1": pred["seqeval"]["overall_f1"],
        "mismatches": dict(Counter(d["tipo"] for d in detalhes)),
        "detalhes": detalhes,
        "resultados_juiz": resultados_juiz,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Métricas do pipeline a partir de JSONL do Doccano.")
    ap.add_argument("gold", help="JSONL de referência (anotação manual)")
    ap.add_argument("pred", nargs="?", help="JSONL a avaliar. Sem ele, treina o NER no gold e avalia")
    ap.add_argument("--labels", help="arquivo com a lista de labels (.json do Doccano ou .txt)")
    ap.add_argument("--juiz", action="store_true", help="treina o juiz (learning-based e estendido)")
    a = ap.parse_args()
    avaliar(a.gold, a.pred, labels=a.labels, usar_juiz=a.juiz)
