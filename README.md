# pipeline_keyphrase_juridico 

## Taxonomia de erro e o Automatic Judge (Nejadgholi et al., 2020)

A base de tudo é essa: em NER, nem todo erro é igual. Errar o span por um token (tipo prever "Banco do" quando o gold é "Banco do Brasil") é um erro bem mais inofensivo do que inventar uma entidade que não existe. O F1 tradicional, porém, trata os dois tipos de erro da mesma forma — ambos são só "errado". O paper resolve isso separando os mismatches em tipos:

- **exato**: span e categoria certos → TP, sem história.
- **Type-1**: o modelo previu uma entidade que não tá no gold → FP.
- **Type-2**: o modelo deixou passar uma entidade do gold → FN.
- **Type-3**: mesmo span, categoria errada.
- **Type-4**: span diferente (mas sobreposto) e categoria errada.
- **Type-5**: span diferente (sobreposto), categoria certa — o caso do "Banco do" / "Banco do Brasil".

Isso tá em `classificar_mismatches`/`classificar_corpus`, em `metricas.py`.

Só que separar os tipos não resolve o problema de decidir quais são "aceitáveis". Pra isso o paper propõe o **Automatic Judge**: um classificador treinado pra reconhecer categoria a partir do texto da entidade (mais uma classe `other` pra "isso nem é entidade"). Pra cada erro Type-3/4/5, ele reclassifica o texto que o modelo previu — se a categoria que o juiz dá bate com a que o NER previu, o erro é aceito; senão, rejeitado. É assim que o pipeline decide, de forma automática, quais desvios de span ou categoria fazem sentido na prática, sem eu ter que arbitrar isso na mão. Tá na classe `Juiz`, em `juiz.py`.

(Duas coisas que vale saber: Type-2 nunca passa pelo juiz, porque não existe texto previsto pra classificar. E Type-3 tende a só confirmar o que já sabia, porque o texto previsto é idêntico ao do gold — o mesmo tipo de texto em que o juiz foi treinado.)

### E é por isso que existem 4 F1s diferentes

Com os tipos e o julgamento em mãos, dá pra montar várias versões do F1, cada uma "perdoando" um conjunto diferente de erros — e é exatamente essa diferença no que é perdoado que explica por que os números saem diferentes:

- **Estrito**: só conta TP o acerto exato. É o F1 de sempre, o mais rígido — qualquer Type-3/4/5 conta como erro em cheio.
- **Relaxado**: todo Type-5 já conta como TP, direto, sem nem passar pelo juiz. Por isso ele é maior que o estrito e o learning-based, e ele perdoa TODO erro de span com categoria certa, mesmo os que fazem pouco sentido.
- **Learning-based** (a proposta do paper): só os Type-5 que o juiz aceitou contam como TP; os que o juiz rejeitou continuam como erro. Por isso ele fica entre o estrito e o relaxado — ele só perdoa a parte dos Type-5 que o juiz validou, não todos.
- **Estendido** (pedido extra do Flávio, não tá no paper): mesma lógica do learning-based, só que também perdoa os Type-3 e Type-4 que o juiz aceitar. Como ele perdoa mais tipos de erro que o learning-based, ele é sempre maior ou igual a ele.

Isso dá a ordem: estrito ≤ learning-based ≤ relaxado sempre, e learning-based ≤ estendido sempre. Entre estendido e relaxado não tem ordem fixa — depende de quantos Type-3/4 o juiz aceitou (que puxa o estendido pra cima) versus quantos Type-5 ele rejeitou (que puxa o relaxado pra cima, já que o relaxado nem olha pro juiz).

Pra dar uma ideia em números: com TPₑ=80, T1=5, T2=5, T3=4, T4=2, T5=9, e o juiz aceitando 1 de cada Type-3/4 e 6 dos 9 Type-5 — estrito sai 0,800, learning-based 0,860, estendido 0,880, relaxado 0,890. Dá pra ver a progressão: cada F1 recupera um pedaço maior do que o anterior, conforme vai perdoando mais erros.

## Métricas de ranking (Hasan & Ng, 2014)

Essas métricas vêm de keyphrase extraction, onde a ordem dos resultados importa muito — o usuário geralmente só olha o topo da lista. O F1, em qualquer variante acima, ignora completamente a ordem, só olha "acertou ou não". Pra trazer essa noção pro NER, uso a confiança do modelo (softmax por token, agregada por média ou mínimo — `confiancas_por_token`/`confianca_span`) como score pra ordenar as entidades previstas de cada sentença (`montar_ranking`).

- **R-precision**: olha só as N primeiras posições do ranking, onde N é quantas entidades o gold tem naquela sentença, e conta quantas são acerto. Exemplo: ranking [a, x, b, c, y], gold {a, b, c} → nas 3 primeiras tem 2 acertos → 2/3 ≈ 0,667. Responde "o topo, do tamanho do gold, tá limpo?".
- **MRR**: olha onde fica o primeiro acerto. Exemplo: ranking [x, y, a], gold {a} → acerto só na posição 3 → RR = 1/3. Responde "quão rápido o usuário acharia a primeira entidade certa?".
- **Bpref**: feita pra quando o gold é incompleto — bem realista aqui, já que ninguém marca 100% das entidades na anotação manual. Para cada gold que aparece no ranking, conta quantos não-gold vieram antes dele. Exemplo: ranking [x, a, y, b, z], gold {a, b} → antes de `a` tem 1 não-gold (1 − 1/2 = 0,5), antes de `b` tem 2 (1 − 2/2 = 0) → Bpref = (0,5+0)/2 = 0,25. Responde "quanto lixo aparece antes dos acertos, mesmo sabendo que o gold pode estar incompleto?".

Implementadas em `r_precision`, `reciprocal_rank`, `bpref`, todas em `metricas.py`, e agregadas por sentença em `avaliar_corpus` (sentenças sem gold são ignoradas na média, não contam como zero).

## Onde está cada coisa

| O que é | Onde está |
|---|---|
| Tipos de erro (Type-1 a 5) | `classificar_mismatches`, `classificar_corpus` — `metricas.py` |
| As 4 variantes de F1 | `calcular_f1s` (usa `f1_por_contagens`) — `metricas.py` |
| Automatic Judge (aceita/rejeita) | classe `Juiz`, método `avaliar_erros` — `juiz.py` |
| R-precision, MRR, Bpref | `r_precision`, `reciprocal_rank`, `bpref`, `avaliar_corpus` — `metricas.py` |
| Confiança e montagem do ranking | `confiancas_por_token`, `confianca_span`, `montar_ranking`, `avaliar_ranking` — `metricas.py` |
| Dados, treino e predição do NER | `ner.py` |


## Referências

- Nejadgholi et al. (2020). *Extensive Error Analysis and a Learning-Based Evaluation of Medical Entity Recognition Systems to Approximate User Experience.* https://aclanthology.org/2020.bionlp-1.19.pdf
- Hasan & Ng (2014). *Automatic Keyphrase Extraction: A Survey of the State of the Art.* https://aclanthology.org/P14-1119.pdf
