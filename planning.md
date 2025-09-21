# Organização das Ideias do Projeto

## 1. Contexto Atual - NQ v0

- **Dataset em uso:** [NQ (Natural Questions)](https://ai.google.com/research/NaturalQuestions).
- **Versão atual:** “Versão 0” → grafo com:

  - Nós = artigos da Wikipedia = 108.071
  - Arestas direcionadas = relações **CITES** = 5.122.983

---

## 2. Objetivos KG4AI

- Realizar análise inicial e exploratória da rede.
- Estudar ferramentas e técnicas de:

  - Pré-processamento e Construção do Grafo
  - Análise Estrutural da Rede
  - Detecção de Comunidades
  - Análise de Homofilia e Vínculos
  - Visualizações e comparações

- **PESQUISA DE MESTRADO** em paralelo:

  - **BUSCAR OPORTUNIDADES!**
  - Palavras-chave: **RAGs**, **Agentes**, **Grandes Grafos de Conhecimento**
  - Meta: manter como “exploração constante”, não sendo foco principal imediato, mas inspirando decisões.

---

## 3. Alinhamento com Redes Complexas - Prof. Ruben

- **Projeto na disciplina:**

  1. Pré-processamento e Construção do Grafo
  2. Análise Estrutural da Rede
  3. Detecção de Comunidades
  4. Análise de Homofilia e Vínculos

- **Observação:** Fortemente relacionado com o que já está sendo desenvolvido com a versão 0 do dataset.

---

## 4. Perguntas/Problemas de Pesquisa em Aberto

- **Como descrever essa rede?**
- **Como comparar diferentes métodos de detecção de comunidades?**
- **Quais visualizações e métricas melhor capturam a estrutura da rede?**

---

## 5. Materiais e Métodos

- **Detecção de comunidades**

  - **Infomap**
  - **Louvain**
  - **Leiden**
  - **K-means ou DBSCAN em embeddings**
  - **HP-MOCD**

- **Avaliação Qualitativa e Métricas**

  - **Condunctance**
  - **Transitividade**
  - **Triad Participation Ratio (TPR)**
  - **Triadic Closure**
  - **Triad Census**
  - **Modularidade**
  - **Clustering Coefficient**
  - **Homofilia**

- **Visualização:**

  - Comparar ML x Algoritmos clássicos de detecção de comunidades.
  - Avaliar clusters obtidos:
    - Métodos de visualização - gráficos, imagens, etc.
    - Analisar métricas estruturais dos clusters.
    - Filtrar **?**.
    - Criar **rede macro** → nós representam comunidades.
      - Grafo ponderado?
      - Grafo com uma constante mínima?
      - Ruben usou 0.3?
    - Usar **LLM para nomear e agrupar comunidades** → **MacroClusters** e **MacroClusters**.
  - Redução de escopo: analisar profundamente algumas comunidades **?**

---

## 6. Futuro / Longo Prazo

- **Renomear relações:** de **CITES** para múltiplos tipos de relações:

  - LLMs - [GAIA **?**](https://deepmind.google/models/gemma/gemmaverse/gaia/);
  - Ideia anti LLM:
    - Lista de Relacionamentos (Vector Database);
    - BERT;
    - Distância Vetores;
    - Uso de comunidades aqui;

- **Construção de grafos de conhecimento:**

  - Curso [Agentic Knowledge Graph Construction](https://learn.deeplearning.ai/my/learnings).
  - Usar LLMs para enriquecer e classificar relações (artigos - AutoSchemaKG, etc).
  - Explorar aplicações em **RAGs** e **Agentes**.
  - Evoluir da versão 0 → versões mais sofisticadas do grafo.

---
