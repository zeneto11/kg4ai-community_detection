# Análise de Nomenclatura de Comunidades (Leiden)
## Avaliação Multi-Métrica: LLM-as-a-Judge + BERT-Score
Relatório gerado em: 2025-11-09 00:43:13
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
- **Score Médio Combinado**: 0.732
- **Score Mediano Combinado**: 0.753
- **Desvio Padrão**: 0.121

### 🤖 Métricas LLM-as-a-Judge
- **Score Médio LLM**: 0.767
- **Score Mediano LLM**: 0.800
- **Desvio Padrão LLM**: 0.185
- **Média do Melhor Match LLM**: 0.825

### 📝 Métricas BERT-Score
- **Precision Média**: 0.509
- **Recall Médio**: 0.334
- **F1 Médio**: 0.679
- **F1 Mediano**: 0.682
- **Desvio Padrão F1**: 0.070

### Distribuição por Threshold - Score Combinado
- **Score >= 0.8**: 6 comunidades (31.58%)
- **Score >= 0.7**: 14 comunidades (73.68%)
- **Score >= 0.6**: 16 comunidades (84.21%)
- **Score >= 0.5**: 17 comunidades (89.47%)

### Distribuição por Threshold - LLM
- **Score >= 0.7**: 16 comunidades (84.21%)
- **Score >= 0.6**: 16 comunidades (84.21%)

### Distribuição por Threshold - BERT F1
- **F1 >= 0.7**: 7 comunidades (36.84%)
- **F1 >= 0.6**: 16 comunidades (84.21%)

### Correlações
- **Correlação LLM vs BERT**: 0.242
- **Correlação Nº Categorias vs Score Combinado**: 0.134
- **Correlação Nº Categorias vs LLM**: 0.052
- **Correlação Nº Categorias vs BERT**: 0.370

### Quartis de Distribuição
#### Score Combinado:
- **25º Percentil**: 0.705
- **50º Percentil (Mediana)**: 0.753
- **75º Percentil**: 0.809

#### LLM Score:
- **25º Percentil**: 0.740
- **50º Percentil (Mediana)**: 0.800
- **75º Percentil**: 0.890

#### BERT F1:
- **25º Percentil**: 0.628
- **50º Percentil (Mediana)**: 0.682
- **75º Percentil**: 0.730

---
## Top 5 Melhores Comunidades (Maior Score Combinado)
### Comunidade 3
**Score Combinado**: 0.905 | **LLM**: 0.972 | **BERT F1**: 0.805
- **Categorias Geradas (LLM)**: `united states history / american states / national & state affairs`
- **Número de Categorias**: 3
- **LLM Scores** (Best: 0.980, Avg: 0.953): 0.980, 0.900, 0.980
- **BERT Scores** (P: 0.519, R: 0.328, F1s): 0.805, 0.529, 0.550
- **Palavras-chave**: _states, united, new, state, act, york, american, national, california, history_
- **Top 5 Categorias de Referência**: 'Legal history of the United States', 'History of United States expansionism', 'Amendments to the United States Constitution', 'States of the United States', 'Wars involving the United States'

### Comunidade 17
**Score Combinado**: 0.869 | **LLM**: 0.920 | **BERT F1**: 0.792
- **Categorias Geradas (LLM)**: `gun laws in u.s. states / concealed carry regulations / california, virginia, carolina`
- **Número de Categorias**: 3
- **LLM Scores** (Best: 0.950, Avg: 0.850): 0.950, 0.800, 0.800
- **BERT Scores** (P: 0.640, R: 0.416, F1s): 0.792, 0.634, 0.494
- **Palavras-chave**: _gun, laws, new, carry, united, states, california, carolina, virginia, concealed_
- **Top 5 Categorias de Referência**: 'United States gun laws by state', 'Texas law', 'Gun politics in the United States', 'Self-defense', 'United States firearms law'

### Comunidade 16
**Score Combinado**: 0.846 | **LLM**: 0.927 | **BERT F1**: 0.723
- **Categorias Geradas (LLM)**: `coronation street characters / british soap opera cast`
- **Número de Categorias**: 2
- **LLM Scores** (Best: 0.950, Avg: 0.875): 0.950, 0.800
- **BERT Scores** (P: 0.625, R: 0.357, F1s): 0.723, 0.650
- **Palavras-chave**: _coronation, street, characters, barlow, connor, platt, baldwin, roache, webster, michelle_
- **Top 5 Categorias de Referência**: 'Coronation Street characters', 'Fictional bartenders', 'Fictional businesspeople', 'Fictional victims of kidnapping', 'Fictional British people'

### Comunidade 10
**Score Combinado**: 0.832 | **LLM**: 0.895 | **BERT F1**: 0.737
- **Categorias Geradas (LLM)**: `sports competitions / american team sports / national seasons`
- **Número de Categorias**: 3
- **LLM Scores** (Best: 0.950, Avg: 0.767): 0.950, 0.950, 0.400
- **BERT Scores** (P: 0.510, R: 0.305, F1s): 0.737, 0.606, 0.553
- **Palavras-chave**: _football, league, basketball, baseball, season, national, major, state, bowl, men_
- **Top 5 Categorias de Referência**: 'Living people', 'National Basketball Association lists', 'National Football League teams', 'Lists of sports championships', 'Major League Baseball teams'

### Comunidade 13
**Score Combinado**: 0.817 | **LLM**: 0.942 | **BERT F1**: 0.628
- **Categorias Geradas (LLM)**: `office workers / tv series characters`
- **Número de Categorias**: 2
- **LLM Scores** (Best: 0.950, Avg: 0.925): 0.900, 0.950
- **BERT Scores** (P: 0.518, R: 0.259, F1s): 0.556, 0.628
- **Palavras-chave**: _office, season, paper, michael, day, night, andy, series, toby, dunder_
- **Top 5 Categorias de Referência**: 'The Office (U.S. TV series) characters', 'Fictional characters introduced in 2005', 'The Office (U.S. TV series) episodes in multiple parts', 'Fictional American people of English descent', 'Fictional receptionists'


---
## Top 5 Piores Comunidades (Menor Score Combinado)
### Comunidade 8
**Score Combinado**: 0.700 | **LLM**: 0.700 | **BERT F1**: 0.700
- **Categorias Geradas (LLM)**: `computer games history / video game software / network systems data`
- **Número de Categorias**: 3
- **LLM Scores** (Best: 0.800, Avg: 0.467): 0.400, 0.800, 0.200
- **BERT Scores** (P: 0.444, R: 0.344, F1s): 0.649, 0.700, 0.600
- **Palavras-chave**: _episodes, computer, video, game, software, system, data, games, history, network_
- **Top 5 Categorias de Referência**: 'Windows games', 'Xbox One games', 'PlayStation 4 games', 'Manga series', 'Xbox 360 games'

### Comunidade 7
**Score Combinado**: 0.687 | **LLM**: 0.760 | **BERT F1**: 0.577
- **Categorias Geradas (LLM)**: `sports & culture / south asia / education & technology`
- **Número de Categorias**: 3
- **LLM Scores** (Best: 0.900, Avg: 0.433): 0.200, 0.900, 0.200
- **BERT Scores** (P: 0.439, R: 0.330, F1s): 0.518, 0.577, 0.530
- **Palavras-chave**: _india, indian, cricket, national, pakistan, institute, film, technology, history, world_
- **Top 5 Categorias de Referência**: 'Indian films', 'Living people', 'Hindi-language films', 'Constitution of India', 'Hindi-language television programs'

### Comunidade 18
**Score Combinado**: 0.586 | **LLM**: 0.540 | **BERT F1**: 0.654
- **Categorias Geradas (LLM)**: `characters' locations / star trek crew`
- **Número de Categorias**: 2
- **LLM Scores** (Best: 0.600, Avg: 0.400): 0.600, 0.200
- **BERT Scores** (P: 0.549, R: 0.299, F1s): 0.654, 0.612
- **Palavras-chave**: _home, away, characters, stewart, braxton, king, casey, darryl, ailsa, kirsty_
- **Top 5 Categorias de Referência**: 'Home and Away characters', 'Lists of Home and Away characters', 'Fictional characters introduced in 1988', 'Fictional businesspeople', 'Fictional teenage parents'

### Comunidade 6
**Score Combinado**: 0.470 | **LLM**: 0.370 | **BERT F1**: 0.621
- **Categorias Geradas (LLM)**: `ancient roman art & religion / catholic history & empire`
- **Número de Categorias**: 2
- **LLM Scores** (Best: 0.400, Avg: 0.300): 0.200, 0.400
- **BERT Scores** (P: 0.593, R: 0.402, F1s): 0.621, 0.617
- **Palavras-chave**: _history, church, roman, ancient, art, god, bible, empire, catholic, book_
- **Top 5 Categorias de Referência**: 'Christian terminology', 'Biblical phrases', 'Former empires', 'Former countries in Europe', 'Former countries in Asia'

### Comunidade 5
**Score Combinado**: 0.447 | **LLM**: 0.290 | **BERT F1**: 0.682
- **Categorias Geradas (LLM)**: `power systems / energy sources / historical vehicles`
- **Número de Categorias**: 3
- **LLM Scores** (Best: 0.300, Avg: 0.267): 0.300, 0.300, 0.200
- **BERT Scores** (P: 0.475, R: 0.348, F1s): 0.668, 0.682, 0.663
- **Palavras-chave**: _power, system, energy, engine, history, ford, solar, series, space, number_
- **Top 5 Categorias de Referência**: 'All-wheel-drive vehicles', '2000s automobiles', '2010s automobiles', 'Rear-wheel-drive vehicles', 'Front-wheel-drive vehicles'


---
## Análise de Distribuição por Número de Categorias

### Comunidades com 2 Categoria(s) (6 total)
- **Score Médio Combinado**: 0.709
- **Score Médio LLM**: 0.753
- **Score Médio BERT F1**: 0.642
- **Melhor Score Combinado**: 0.846
- **Pior Score Combinado**: 0.470

### Comunidades com 3 Categoria(s) (13 total)
- **Score Médio Combinado**: 0.743
- **Score Médio LLM**: 0.774
- **Score Médio BERT F1**: 0.696
- **Melhor Score Combinado**: 0.905
- **Pior Score Combinado**: 0.447

---
## Tabela Completa de Resultados

| ID | Categorias | # Cat | Score Comb | LLM Score | BERT F1 | BERT Prec | BERT Rec | LLM Scores | BERT F1s | Top 5 Categorias |
|---:|:-----------|------:|-----------:|----------:|--------:|----------:|---------:|:-----------|:---------|:---------------|
| 0 | film & tv content / actors & characters | 2 | 0.764 | 0.885 | 0.583 | 0.468 | 0.310 | 0.900, 0.800 | 0.583, 0.539 | 'English-language films', 'American films', 'English-language television programs', 'Living people', 'IMAX films' |
| 1 | music & album / love song | 2 | 0.770 | 0.855 | 0.644 | 0.407 | 0.252 | 0.900, 0.600 | 0.479, 0.644 | 'Billboard Hot 100 number-one singles', 'UK Singles Chart number-one singles', 'Number-one singles in Australia', 'Irish Singles Chart number-one singles', 'RPM Top Singles number-one singles' |
| 2 | global affairs / geography / sports & competitions | 3 | 0.800 | 0.820 | 0.771 | 0.504 | 0.336 | 0.900, 0.800, 0.200 | 0.771, 0.584, 0.555 | 'Member states of the United Nations', 'Visa requirements by nationality', 'Living people', 'Lists of countries', 'English-speaking countries and territories' |
| 3 | united states history / american states / national & state affairs | 3 | 0.905 | 0.972 | 0.805 | 0.519 | 0.328 | 0.980, 0.900, 0.980 | 0.805, 0.529, 0.550 | 'Legal history of the United States', 'History of United States expansionism', 'Amendments to the United States Constitution', 'States of the United States', 'Wars involving the United States' |
| 4 | medical history / american cuisine / human health | 3 | 0.710 | 0.720 | 0.694 | 0.428 | 0.335 | 0.600, 0.200, 0.800 | 0.682, 0.554, 0.694 | 'Wikipedia articles incorporating text from the 20th edition of Gray&#39;s Anatomy (1918)', 'Metabolism', 'IUCN Red List least concern species', 'Cellular respiration', 'Digestive system' |
| 5 | power systems / energy sources / historical vehicles | 3 | 0.447 | 0.290 | 0.682 | 0.475 | 0.348 | 0.300, 0.300, 0.200 | 0.668, 0.682, 0.663 | 'All-wheel-drive vehicles', '2000s automobiles', '2010s automobiles', 'Rear-wheel-drive vehicles', 'Front-wheel-drive vehicles' |
| 6 | ancient roman art & religion / catholic history & empire | 2 | 0.470 | 0.370 | 0.621 | 0.593 | 0.402 | 0.200, 0.400 | 0.621, 0.617 | 'Christian terminology', 'Biblical phrases', 'Former empires', 'Former countries in Europe', 'Former countries in Asia' |
| 7 | sports & culture / south asia / education & technology | 3 | 0.687 | 0.760 | 0.577 | 0.439 | 0.330 | 0.200, 0.900, 0.200 | 0.518, 0.577, 0.530 | 'Indian films', 'Living people', 'Hindi-language films', 'Constitution of India', 'Hindi-language television programs' |
| 8 | computer games history / video game software / network systems data | 3 | 0.700 | 0.700 | 0.700 | 0.444 | 0.344 | 0.400, 0.800, 0.200 | 0.649, 0.700, 0.600 | 'Windows games', 'Xbox One games', 'PlayStation 4 games', 'Manga series', 'Xbox 360 games' |
| 9 | social sciences / legal studies / economic & management theory | 3 | 0.753 | 0.760 | 0.743 | 0.518 | 0.422 | 0.800, 0.600, 0.600 | 0.711, 0.743, 0.691 | 'Business terms', 'Production economics', 'Contract law', 'Legal doctrines and principles', 'Sociological theories' |
| 10 | sports competitions / american team sports / national seasons | 3 | 0.832 | 0.895 | 0.737 | 0.510 | 0.305 | 0.950, 0.950, 0.400 | 0.737, 0.606, 0.553 | 'Living people', 'National Basketball Association lists', 'National Football League teams', 'Lists of sports championships', 'Major League Baseball teams' |
| 11 | football community / united kingdom / fifa matters | 3 | 0.767 | 0.860 | 0.627 | 0.455 | 0.322 | 0.800, 0.900, 0.600 | 0.592, 0.624, 0.627 | 'Surnames', 'Countries at the FIFA World Cup', '2018 FIFA World Cup', 'FIFA World Cup tournaments', 'Living people' |
| 12 | gambling & entertainment / las vegas / maritime | 3 | 0.716 | 0.800 | 0.590 | 0.479 | 0.318 | 0.400, 0.900, 0.400 | 0.590, 0.572, 0.549 | 'Fast-food chains of the United States', 'Fast-food franchises', 'Las Vegas Strip', 'RMS Titanic', 'Restaurant chains in the United States' |
| 13 | office workers / tv series characters | 2 | 0.817 | 0.942 | 0.628 | 0.518 | 0.259 | 0.900, 0.950 | 0.556, 0.628 | 'The Office (U.S. TV series) characters', 'Fictional characters introduced in 2005', 'The Office (U.S. TV series) episodes in multiple parts', 'Fictional American people of English descent', 'Fictional receptionists' |
| 14 | "days of our lives characters" / "horton-dimera family" / "carver-kiriakis clan" | 3 | 0.719 | 0.780 | 0.628 | 0.521 | 0.322 | 0.900, 0.400, 0.200 | 0.628, 0.481, 0.536 | 'Days of Our Lives characters', 'Living people', 'Days of Our Lives', 'Fictional American people of Irish descent', 'Fictional American people of English descent' |
| 15 | "nursery scene" / "urban landscape" / "weather event" | 3 | 0.751 | 0.780 | 0.707 | 0.584 | 0.346 | 0.900, 0.300, 0.300 | 0.707, 0.579, 0.573 | 'Roud Folk Song Index songs', 'English nursery rhymes', 'Children&#39;s songs', 'Year of song unknown', 'Songwriter unknown' |
| 16 | coronation street characters / british soap opera cast | 2 | 0.846 | 0.927 | 0.723 | 0.625 | 0.357 | 0.950, 0.800 | 0.723, 0.650 | 'Coronation Street characters', 'Fictional bartenders', 'Fictional businesspeople', 'Fictional victims of kidnapping', 'Fictional British people' |
| 17 | gun laws in u.s. states / concealed carry regulations / california, virginia, carolina | 3 | 0.869 | 0.920 | 0.792 | 0.640 | 0.416 | 0.950, 0.800, 0.800 | 0.792, 0.634, 0.494 | 'United States gun laws by state', 'Texas law', 'Gun politics in the United States', 'Self-defense', 'United States firearms law' |
| 18 | characters' locations / star trek crew | 2 | 0.586 | 0.540 | 0.654 | 0.549 | 0.299 | 0.600, 0.200 | 0.654, 0.612 | 'Home and Away characters', 'Lists of Home and Away characters', 'Fictional characters introduced in 1988', 'Fictional businesspeople', 'Fictional teenage parents' |