# MADS-ML-ReneS

**Auteur:** René Santifort  
**Studentnummer:** 1918885  
**Cursus:** Machine Learning (TMOI-ML-20)  
**Datum:** 27 januari 2026

---

## Projectomschrijving

Dit project bevat het winnende model van de eindopdracht voor de cursus Machine Learning (MADS).  
Het doel is het automatisch classificeren van juridische akten naar type (32 klassen), op basis van de tekstinhoud van notariële documenten.

Het beste model combineert twee BERT-varianten (Legal Dutch BERT en Legal BERT) met een Regex-vectorizer en een skip-verbinding, en behaalt de hoogste validatie-F1(micro)-score van alle geteste architecturen.

---

## Repository structuur

```
MADS-ML-ReneS/
├── h6_run2.py      # Winnend model: 2xBERT + Regex + MeanAggregator + Skip
└── README.md
```

---

## Model architectuur

Het winnende model (`h6_run2.py`) bestaat uit de volgende componenten:

- **Twee parallelle BERT-takken:**
  - `BERT (legal_dutch)` — Nederlandse juridische embeddings
  - `BERT (legal_bert)` — Engelstalige Legal BERT embeddings
- **Regex-vectorizer:** detecteert juridische referenties (wetsartikelen) als extra feature
- **MeanAggregator:** aggregeert chunk-embeddings per document
- **Parallel + Concatenate2D:** combineert de drie invoerstromen
- **Projection layer:** comprimeert naar 32 klassen
- **Skip-verbinding:** bewaart harde signalen en verbetert generalisatie

De pipeline in vereenvoudigde vorm:

```
Parallel(
    Serial([MeanAggregator, NeuralNet(768→32)])   ← legal_dutch
    Serial([MeanAggregator, NeuralNet(768→32)])   ← legal_bert
    Serial([NeuralNet(REGEX_DIM→32)])              ← regex features
)
→ Concatenate2D
→ Projection(96→32)
→ Skip(NeuralNet(32→32))
```

---

## Resultaten

| F1 (micro) | Run | Input | Aggregator | Architectuur | Head |
|---|---|---|---|---|---|
| 0.881 | 1 | BERT(legal_dutch) + Regex | Mean (BERT path) | Parallel | MLP |
| **0.901** | **2 (winnaar)** | **BERT(legal_dutch) + BERT(legal_bert) + Regex** | **Mean (all berts)** | **Parallel** | **Projection + Skip** |

Het winnende model (run 2) behaalt een **F1(micro) van ~0.901** op de validatieset.

---

## Hypothesen

Het experiment is opgebouwd rondom 5 hypothesen:

1. **AttentionAggregator vs MeanAggregator** — Attention zou betere focus op relevante chunks moeten geven.
2. **Padding-strategie en max_chunks** — Het variëren van `max_chunks` en gebruik van Dynamic Padding zoekt een "sweet spot" in de F1-score.
3. **Skip-verbindingen** — Behoudt harde signalen zoals regex-features tegen vervorming tijdens training.
4. **Combinatie Legal BERT + Legal Dutch** — Twee BERT-varianten samen geven betere representaties dan één model.
5. **Uitbreiding classificatie-head** — Groter netwerk met dropout en normalisatie zou overfitting verminderen.

Opvallend resultaat: de `MeanAggregator` presteerde uiteindelijk beter dan de `AttentionAggregator`. De skip-verbinding en de combinatie van twee BERT-modellen bleken de meest impactvolle verbeteringen.

---

## Installatie en gebruik

### Vereisten

- Python 3.10+
- PyTorch (met CUDA-ondersteuning aanbevolen)
- [`mltrainer`](https://github.com/raoulg/mltrainer)
- [`vectormesh`](https://github.com/raoulg/vectormesh)
- `pydantic`

### Data

Zorg dat de voorbewerkte vector-caches beschikbaar zijn in de map `../artefacts/`, met de volgende naamconventie:

```
../artefacts/*legal_dutch*train/
../artefacts/*legal_dutch*valid/
../artefacts/*legal_bert*train/
../artefacts/*legal_bert*valid/
```

### Uitvoeren

Voer het script uit vanaf de terminal:

```bash
python h6_run2.py
```

Het script detecteert automatisch GPU/CPU, laadt de caches, bouwt de Regex-vectorizer, en start de trainingsloop. Resultaten worden opgeslagen in `demo/hypothese6_run2_mean_skip_2bert_plus_regex/` als TensorBoard-logs en TOML-bestanden.

---

## Trainingsinstellingen

| Parameter | Waarde |
|---|---|
| Epochs | 50 |
| Batch size | 32 |
| Optimizer | Adam |
| Scheduler | ReduceLROnPlateau |
| Loss functie | BCEWithLogitsLoss |
| Chunks (padding) | 30 (fixed) |
| Regex max features | 200 |
| Klassen | 32 |
| Metrics | F1(micro), F1(macro) |

---

## Eindconclusie

Het beste resultaat werd niet behaald door één groot taalmodel, maar door een combinatie van bronnen en technieken: Nederlandse en Engelse BERT-embeddings gecombineerd met juridische regex-features, doorgegeven via een skip-verbinding. De `regex_dim` past zich automatisch aan op het aantal gevonden wetsartikelen in de trainingsdata, wat het model robuust maakt voor variatie in documentlengte.
