# pipeline_keyphrase_juridico

Avaliação de NER (BERTimbau + LeNER-Br) com taxonomia de mismatches, Automatic Judge e métricas de ranking.
Projeto de IC — CIn-UFPE.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `pipeline_keyphrase_juridico.ipynb` | Fluxo completo (treino do NER, treino do juiz, avaliação) — roda de cima para baixo |
| `entidades.py` | BIO → entidades, overlap de spans, classificação de mismatches (Type-1 a 5) |
| `juiz.py` | Candidatos `other` (noun chunks spaCy) e avaliação dos erros pelo Automatic Judge |
| `metricas.py` | F1 por contagens, R-precision, MRR, Bpref, confiança do NER e ranking |

## Métricas

- **F1 estrito / relaxado / learning-based / estendido** — calculados no notebook a partir das contagens de mismatches (`f1_por_contagens`). Learning-based = só Type-5 aceitos pelo juiz deixam de ser penalizados; estendido = Types 3, 4 e 5 aceitos.
- **R-precision, MRR, Bpref** — `metricas.py`; score de ranking = confiança do NER por span (média ou mínimo dos tokens).
- Type-2 não é avaliável pelo juiz (não há span previsto).

## Referências

- Nejadgholi et al. (2020). *Extensive Error Analysis and a Learning-Based Evaluation of Medical Entity Recognition Systems to Approximate User Experience.* https://aclanthology.org/2020.bionlp-1.19.pdf
- Hasan & Ng (2014). *Automatic Keyphrase Extraction: A Survey of the State of the Art.* https://aclanthology.org/P14-1119.pdf

## Como rodar (Colab)

```python
!git clone https://github.com/SEU_USUARIO/SEU_REPO.git
%cd SEU_REPO
```

Abra o notebook e execute as células em ordem (GPU recomendada). Dependências: `transformers`, `datasets`, `evaluate`, `seqeval`, `spacy` (`pt_core_news_lg`), `scikit-learn`, `scipy`, `torch`.
