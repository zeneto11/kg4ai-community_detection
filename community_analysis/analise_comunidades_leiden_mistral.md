# Análise de Nomenclatura de Comunidades (Leiden)
## Avaliação Multi-Métrica: LLM-as-a-Judge + BERT-Score
Relatório gerado em: 2025-10-29 17:39:33
Modelo LLM: mistral
Modelo BERT: microsoft/deberta-xlarge-mnli

**Nota**: As avaliações comparam as categorias geradas pelo LLM com as keywords extraídas das comunidades.

---
## Resumo Executivo das Métricas
- **Total de Comunidades Analisadas**: 19
- **Média de Categorias por Comunidade**: 2.68
  - Comunidades com 1 categoria: 0
  - Comunidades com 2 categorias: 6
  - Comunidades com 3 categorias: 13

### 📊 Métricas de Score Combinado (60% LLM + 40% BERT)
- **Score Médio Combinado**: 0.734
- **Score Mediano Combinado**: 0.763
- **Desvio Padrão**: 0.094

### 🤖 Métricas LLM-as-a-Judge
- **Score Médio LLM**: 0.777
- **Score Mediano LLM**: 0.800
- **Desvio Padrão LLM**: 0.137
- **Média do Melhor Match LLM**: 0.800

### 📝 Métricas BERT-Score
- **Precision Média**: 0.558
- **Recall Médio**: 0.430
- **F1 Médio**: 0.670
- **F1 Mediano**: 0.681
- **Desvio Padrão F1**: 0.062

### Distribuição por Threshold - Score Combinado
- **Score >= 0.8**: 4 comunidades (21.05%)
- **Score >= 0.7**: 15 comunidades (78.95%)
- **Score >= 0.6**: 18 comunidades (94.74%)
- **Score >= 0.5**: 18 comunidades (94.74%)

### Distribuição por Threshold - LLM
- **Score >= 0.7**: 16 comunidades (84.21%)
- **Score >= 0.6**: 17 comunidades (89.47%)

### Distribuição por Threshold - BERT F1
- **F1 >= 0.7**: 7 comunidades (36.84%)
- **F1 >= 0.6**: 17 comunidades (89.47%)

### Correlações
- **Correlação LLM vs BERT**: 0.377
- **Correlação Nº Categorias vs Score Combinado**: -0.159
- **Correlação Nº Categorias vs LLM**: -0.166
- **Correlação Nº Categorias vs BERT**: -0.056

### Quartis de Distribuição
#### Score Combinado:
- **25º Percentil**: 0.706
- **50º Percentil (Mediana)**: 0.763
- **75º Percentil**: 0.793

#### LLM Score:
- **25º Percentil**: 0.750
- **50º Percentil (Mediana)**: 0.800
- **75º Percentil**: 0.870

#### BERT F1:
- **25º Percentil**: 0.634
- **50º Percentil (Mediana)**: 0.681
- **75º Percentil**: 0.718

---
## Top 5 Melhores Comunidades (Maior Score Combinado)
### Comunidade 3
**Score Combinado**: 0.832 | **LLM**: 0.900 | **BERT F1**: 0.730
- **Categorias Geradas (LLM)**: `united states history / american states / national & state affairs`
- **Número de Categorias**: 3
- **LLM Scores** (Best: 0.900, Avg: 0.900): 0.900, 0.900, 0.900
- **BERT Scores** (P: 0.569, R: 0.442, F1s): 0.612, 0.730, 0.499
- **Palavras-chave**: _states, united, new, state, act, york, american, national, california, history_
- **Top 5 Keywords de Referência**: 'states', 'united', 'new', 'state', 'act'

### Comunidade 0
**Score Combinado**: 0.817 | **LLM**: 0.900 | **BERT F1**: 0.693
- **Categorias Geradas (LLM)**: `film & tv content / actors & characters`
- **Número de Categorias**: 2
- **LLM Scores** (Best: 0.900, Avg: 0.900): 0.900, 0.900
- **BERT Scores** (P: 0.573, R: 0.486, F1s): 0.497, 0.693
- **Palavras-chave**: _film, series, season, episodes, characters, man, you, actor, star, show_
- **Top 5 Keywords de Referência**: 'film', 'series', 'season', 'episodes', 'characters'

### Comunidade 16
**Score Combinado**: 0.812 | **LLM**: 0.900 | **BERT F1**: 0.681
- **Categorias Geradas (LLM)**: `coronation street characters / british soap opera cast`
- **Número de Categorias**: 2
- **LLM Scores** (Best: 0.900, Avg: 0.900): 0.900, 0.900
- **BERT Scores** (P: 0.651, R: 0.419, F1s): 0.681, 0.557
- **Palavras-chave**: _coronation, street, characters, barlow, connor, platt, baldwin, roache, webster, michelle_
- **Top 5 Keywords de Referência**: 'coronation', 'street', 'characters', 'barlow', 'connor'

### Comunidade 17
**Score Combinado**: 0.809 | **LLM**: 0.880 | **BERT F1**: 0.703
- **Categorias Geradas (LLM)**: `gun laws in u.s. states / concealed carry regulations / california, virginia, carolina`
- **Número de Categorias**: 3
- **LLM Scores** (Best: 0.900, Avg: 0.833): 0.800, 0.900, 0.800
- **BERT Scores** (P: 0.696, R: 0.516, F1s): 0.477, 0.660, 0.703
- **Palavras-chave**: _gun, laws, new, carry, united, states, california, carolina, virginia, concealed_
- **Top 5 Keywords de Referência**: 'gun', 'laws', 'new', 'carry', 'united'

### Comunidade 9
**Score Combinado**: 0.794 | **LLM**: 0.860 | **BERT F1**: 0.694
- **Categorias Geradas (LLM)**: `social sciences / legal studies / economic & management theory`
- **Número de Categorias**: 3
- **LLM Scores** (Best: 0.900, Avg: 0.767): 0.800, 0.900, 0.600
- **BERT Scores** (P: 0.554, R: 0.454, F1s): 0.694, 0.630, 0.612
- **Palavras-chave**: _law, theory, social, management, analysis, history, psychology, market, model, tax_
- **Top 5 Keywords de Referência**: 'law', 'theory', 'social', 'management', 'analysis'


---
## Top 5 Piores Comunidades (Menor Score Combinado)
### Comunidade 5
**Score Combinado**: 0.704 | **LLM**: 0.740 | **BERT F1**: 0.651
- **Categorias Geradas (LLM)**: `power systems / energy sources / historical vehicles`
- **Número de Categorias**: 3
- **LLM Scores** (Best: 0.800, Avg: 0.600): 0.600, 0.800, 0.400
- **BERT Scores** (P: 0.540, R: 0.432, F1s): 0.651, 0.526, 0.556
- **Palavras-chave**: _power, system, energy, engine, history, ford, solar, series, space, number_
- **Top 5 Keywords de Referência**: 'power', 'system', 'energy', 'engine', 'history'

### Comunidade 13
**Score Combinado**: 0.694 | **LLM**: 0.770 | **BERT F1**: 0.580
- **Categorias Geradas (LLM)**: `office workers / tv series characters`
- **Número de Categorias**: 2
- **LLM Scores** (Best: 0.800, Avg: 0.700): 0.600, 0.800
- **BERT Scores** (P: 0.442, R: 0.355, F1s): 0.580, 0.492
- **Palavras-chave**: _office, season, paper, michael, day, night, andy, series, toby, dunder_
- **Top 5 Keywords de Referência**: 'office', 'season', 'paper', 'michael', 'day'

### Comunidade 18
**Score Combinado**: 0.652 | **LLM**: 0.600 | **BERT F1**: 0.730
- **Categorias Geradas (LLM)**: `characters' locations / star trek crew`
- **Número de Categorias**: 2
- **LLM Scores** (Best: 0.600, Avg: 0.600): 0.600, 0.600
- **BERT Scores** (P: 0.502, R: 0.375, F1s): 0.730, 0.499
- **Palavras-chave**: _home, away, characters, stewart, braxton, king, casey, darryl, ailsa, kirsty_
- **Top 5 Keywords de Referência**: 'home', 'away', 'characters', 'stewart', 'braxton'

### Comunidade 4
**Score Combinado**: 0.644 | **LLM**: 0.580 | **BERT F1**: 0.740
- **Categorias Geradas (LLM)**: `medical history / american cuisine / human health`
- **Número de Categorias**: 3
- **LLM Scores** (Best: 0.600, Avg: 0.533): 0.600, 0.400, 0.600
- **BERT Scores** (P: 0.515, R: 0.402, F1s): 0.672, 0.740, 0.581
- **Palavras-chave**: _cell, history, human, cuisine, system, disease, food, medical, american, muscle_
- **Top 5 Keywords de Referência**: 'cell', 'history', 'human', 'cuisine', 'system'

### Comunidade 15
**Score Combinado**: 0.419 | **LLM**: 0.360 | **BERT F1**: 0.506
- **Categorias Geradas (LLM)**: `"nursery scene" / "urban landscape" / "weather event"`
- **Número de Categorias**: 3
- **LLM Scores** (Best: 0.400, Avg: 0.267): 0.400, 0.200, 0.200
- **BERT Scores** (P: 0.542, R: 0.408, F1s): 0.467, 0.506, 0.477
- **Palavras-chave**: _little, nyah, old, baby, mary, rain, down, man, row, round_
- **Top 5 Keywords de Referência**: 'little', 'nyah', 'old', 'baby', 'mary'


---
## Análise de Distribuição por Número de Categorias

### Comunidades com 2 Categoria(s) (6 total)
- **Score Médio Combinado**: 0.755
- **Score Médio LLM**: 0.809
- **Score Médio BERT F1**: 0.674
- **Melhor Score Combinado**: 0.817
- **Pior Score Combinado**: 0.652

### Comunidades com 3 Categoria(s) (13 total)
- **Score Médio Combinado**: 0.724
- **Score Médio LLM**: 0.762
- **Score Médio BERT F1**: 0.667
- **Melhor Score Combinado**: 0.832
- **Pior Score Combinado**: 0.419

---
## Tabela Completa de Resultados

| ID | Categorias | # Cat | Score Comb | LLM Score | BERT F1 | BERT Prec | BERT Rec | LLM Scores | BERT F1s | Top 5 Keywords |
|---:|:-----------|------:|-----------:|----------:|--------:|----------:|---------:|:-----------|:---------|:---------------|
| 0 | film & tv content / actors & characters | 2 | 0.817 | 0.900 | 0.693 | 0.573 | 0.486 | 0.900, 0.900 | 0.497, 0.693 | 'film', 'series', 'season', 'episodes', 'characters' |
| 1 | music & album / love song | 2 | 0.763 | 0.800 | 0.708 | 0.573 | 0.394 | 0.800, 0.800 | 0.648, 0.708 | 'song', 'you', 'love', 'album', 'music' |
| 2 | global affairs / geography / sports & competitions | 3 | 0.709 | 0.760 | 0.631 | 0.494 | 0.380 | 0.800, 0.600, 0.600 | 0.583, 0.631, 0.505 | 'world', 'war', 'south', 'united', 'history' |
| 3 | united states history / american states / national & state affairs | 3 | 0.832 | 0.900 | 0.730 | 0.569 | 0.442 | 0.900, 0.900, 0.900 | 0.612, 0.730, 0.499 | 'states', 'united', 'new', 'state', 'act' |
| 4 | medical history / american cuisine / human health | 3 | 0.644 | 0.580 | 0.740 | 0.515 | 0.402 | 0.600, 0.400, 0.600 | 0.672, 0.740, 0.581 | 'cell', 'history', 'human', 'cuisine', 'system' |
| 5 | power systems / energy sources / historical vehicles | 3 | 0.704 | 0.740 | 0.651 | 0.540 | 0.432 | 0.600, 0.800, 0.400 | 0.651, 0.526, 0.556 | 'power', 'system', 'energy', 'engine', 'history' |
| 6 | ancient roman art & religion / catholic history & empire | 2 | 0.793 | 0.885 | 0.654 | 0.640 | 0.508 | 0.800, 0.900 | 0.631, 0.654 | 'history', 'church', 'roman', 'ancient', 'art' |
| 7 | sports & culture / south asia / education & technology | 3 | 0.709 | 0.760 | 0.632 | 0.535 | 0.420 | 0.600, 0.800, 0.600 | 0.492, 0.632, 0.574 | 'india', 'indian', 'cricket', 'national', 'pakistan' |
| 8 | computer games history / video game software / network systems data | 3 | 0.723 | 0.800 | 0.607 | 0.634 | 0.485 | 0.800, 0.800, 0.800 | 0.607, 0.596, 0.569 | 'episodes', 'computer', 'video', 'game', 'software' |
| 9 | social sciences / legal studies / economic & management theory | 3 | 0.794 | 0.860 | 0.694 | 0.554 | 0.454 | 0.800, 0.900, 0.600 | 0.694, 0.630, 0.612 | 'law', 'theory', 'social', 'management', 'analysis' |
| 10 | sports competitions / american team sports / national seasons | 3 | 0.770 | 0.860 | 0.636 | 0.465 | 0.381 | 0.900, 0.800, 0.600 | 0.582, 0.515, 0.636 | 'football', 'league', 'basketball', 'baseball', 'season' |
| 11 | football community / united kingdom / fifa matters | 3 | 0.781 | 0.800 | 0.752 | 0.492 | 0.399 | 0.800, 0.800, 0.800 | 0.651, 0.635, 0.752 | 'cup', 'surname', 'united', 'world', 'fifa' |
| 12 | gambling & entertainment / las vegas / maritime | 3 | 0.735 | 0.740 | 0.728 | 0.526 | 0.407 | 0.800, 0.800, 0.200 | 0.601, 0.728, 0.612 | 'las', 'vegas', 'card', 'casino', 'poker' |
| 13 | office workers / tv series characters | 2 | 0.694 | 0.770 | 0.580 | 0.442 | 0.355 | 0.600, 0.800 | 0.580, 0.492 | 'office', 'season', 'paper', 'michael', 'day' |
| 14 | "days of our lives characters" / "horton-dimera family" / "carver-kiriakis clan" | 3 | 0.782 | 0.860 | 0.665 | 0.660 | 0.510 | 0.900, 0.800, 0.600 | 0.567, 0.665, 0.650 | 'brady', 'days', 'our', 'lives', 'dimera' |
| 15 | "nursery scene" / "urban landscape" / "weather event" | 3 | 0.419 | 0.360 | 0.506 | 0.542 | 0.408 | 0.400, 0.200, 0.200 | 0.467, 0.506, 0.477 | 'little', 'nyah', 'old', 'baby', 'mary' |
| 16 | coronation street characters / british soap opera cast | 2 | 0.812 | 0.900 | 0.681 | 0.651 | 0.419 | 0.900, 0.900 | 0.681, 0.557 | 'coronation', 'street', 'characters', 'barlow', 'connor' |
| 17 | gun laws in u.s. states / concealed carry regulations / california, virginia, carolina | 3 | 0.809 | 0.880 | 0.703 | 0.696 | 0.516 | 0.800, 0.900, 0.800 | 0.477, 0.660, 0.703 | 'gun', 'laws', 'new', 'carry', 'united' |
| 18 | characters' locations / star trek crew | 2 | 0.652 | 0.600 | 0.730 | 0.502 | 0.375 | 0.600, 0.600 | 0.730, 0.499 | 'home', 'away', 'characters', 'stewart', 'braxton' |