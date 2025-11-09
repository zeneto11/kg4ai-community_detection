import json
import pandas as pd
import numpy as np
import requests
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import logging
from bert_score import score as bert_score
from typing import List, Dict, Tuple
import torch

# --- Configuração ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Arquivos de entrada
COMMUNITIES_FILE = 'run142_raw_communities.json'
NODES_FILE = 'nodes.json'
KEYWORDS_FILE = 'run142_community_keywords.json'

# Arquivo de saída
REPORT_FILE = 'analise_comunidades_leiden_mistral.md'

# Configurações do Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"
OLLAMA_TIMEOUT = 300 # 5 minutos por request
MAX_WORKERS = 1 # Ajuste conforme a capacidade da sua máquina

# Configurações do BERT-Score
BERT_MODEL = "microsoft/deberta-xlarge-mnli"  # Modelo de alta qualidade para inglês
BERT_LANG = "en"  # Língua dos textos
USE_FAST_TOKENIZER = True

# Configurar device para GPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    logging.info(f"🚀 GPU detectada: {torch.cuda.get_device_name(0)}")
    logging.info(f"   CUDA version: {torch.version.cuda}")
    logging.info(f"   Memória GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    logging.warning("⚠️  GPU não disponível, usando CPU")

# --- Funções de Interação com LLM ---

def call_ollama(prompt, model=OLLAMA_MODEL):
    """Função genérica para chamar a API do Ollama."""
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"seed": 42, "temperature": 0}
    }
    headers = {"Content-Type": "application/json"}
    try:
        resp = requests.post(OLLAMA_URL, headers=headers, json=data, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
        result = resp.json()
        return result.get("response", "").strip().lower()
    except requests.exceptions.RequestException as e:
        logging.error(f"Erro na chamada da API Ollama: {e}")
        return "erro_api"

def categorize_with_ollama(keywords_string):
    """Gera 1-3 nomes de categorias para uma lista de palavras-chave."""
    prompt = f"""
You are a precise classifier. Analyze the following keywords produced by a community detection algorithm and identify the main themes.

Keywords:
{keywords_string}

Task: Generate between 1 and 3 category names that best describe the community. Each category should be 2-4 words maximum.
- Use 1 category if the keywords represent a single coherent theme
- Use 2-3 categories if multiple distinct but related themes are present

Return ONLY the category names separated by semicolons (;). No explanations, no numbering, no extra text.
Example format: "machine learning; artificial intelligence" or "biology"
"""
    response = call_ollama(prompt)
    # Processar a resposta para extrair as categorias
    if response and response != "erro_api":
        categories = [cat.strip() for cat in response.split(';') if cat.strip()]
        # Limitar a 3 categorias no máximo
        return categories[:3] if categories else ["categoria não identificada"]
    return ["erro_api"]

def calculate_bert_score(categories_llm: List[str], top_categories: List[Tuple[str, int]]) -> Dict[str, float]:
    """
    Calcula BERT-Score entre as categorias LLM e as top keywords de referência.
    Retorna precision, recall e F1 scores.
    """
    if not top_categories or not categories_llm:
        return {
            "bert_precision": 0.0,
            "bert_recall": 0.0,
            "bert_f1": 0.0,
            "bert_individual_f1": []
        }
    
    # Log do device sendo usado (apenas na primeira chamada)
    if not hasattr(calculate_bert_score, '_logged'):
        logging.info(f"💻 BERT-Score usando: {DEVICE.upper()}")
        if DEVICE == 'cuda':
            logging.info(f"   Modelo: {BERT_MODEL}")
            logging.info(f"   Batch size: 64 (otimizado para GPU)")
        calculate_bert_score._logged = True
    
    # Preparar strings de referência (top keywords)
    references = [cat for cat, _ in top_categories[:20]]
    
    # Para cada categoria LLM, calcular o melhor match com as referências
    individual_f1_scores = []
    
    try:
        for cat_llm in categories_llm:
            # Calcular BERT-Score desta categoria contra todas as referências
            P, R, F1 = bert_score(
                [cat_llm] * len(references),  # candidatos (repetidos)
                references,  # referências
                model_type=BERT_MODEL,
                lang=BERT_LANG,
                verbose=False,
                device=DEVICE,
                batch_size=64  # Tamanho de batch otimizado para GPU
            )
            
            # Pegar o melhor F1 score (melhor match com qualquer categoria de referência)
            best_f1 = float(F1.max().item())
            individual_f1_scores.append(best_f1)
        
        # Calcular métricas agregadas
        bert_f1 = max(individual_f1_scores) if individual_f1_scores else 0.0
        
        # Calcular precision e recall médios (usando todas as categorias LLM)
        all_candidates = categories_llm
        all_references = [", ".join(references)] * len(all_candidates)
        
        P, R, F1 = bert_score(
            all_candidates,
            all_references,
            model_type=BERT_MODEL,
            lang=BERT_LANG,
            verbose=False,
            device=DEVICE,
            batch_size=64
        )
        
        bert_precision = float(P.mean().item())
        bert_recall = float(R.mean().item())
        
    except Exception as e:
        logging.error(f"Erro ao calcular BERT-Score: {e}")
        return {
            "bert_precision": 0.0,
            "bert_recall": 0.0,
            "bert_f1": 0.0,
            "bert_individual_f1": []
        }
    
    return {
        "bert_precision": bert_precision,
        "bert_recall": bert_recall,
        "bert_f1": bert_f1,
        "bert_individual_f1": individual_f1_scores
    }


def evaluate_correlation_with_ollama(categories_llm, top_categories):
    """Avalia a correlação semântica usando LLM-as-a-Judge e retorna múltiplos scores."""
    if not top_categories:
        return {"overall_score": 0.0, "individual_scores": [], "best_match": 0.0, "avg_score": 0.0}
    
    top_cats_str = ", ".join([cat for cat, _ in top_categories[:20]])
    
    # Se categories_llm é uma string, converter para lista
    if isinstance(categories_llm, str):
        categories_llm = [cat.strip() for cat in categories_llm.split(';') if cat.strip()]
    
    individual_scores = []
    
    # Avaliar cada categoria LLM individualmente
    for cat_llm in categories_llm:
        prompt = f"""
You are an accurate semantic evaluator. Evaluate the relevance of the following category created by an LLM model relative to a community's top keywords.

LLM-generated category: "{cat_llm}"
Top keywords from the community: "{top_cats_str}"

Your task is to provide a single floating-point number between 0.0 and 1.0, where:
- 0.0 = no semantic relationship
- 0.3 = weak/tangential relationship
- 0.5 = moderate relationship
- 0.7 = strong relationship
- 1.0 = perfect semantic match

Return ONLY the number, nothing else.
"""
        response = call_ollama(prompt)
        try:
            score = float(response)
            individual_scores.append(score)
        except (ValueError, TypeError):
            logging.warning(f"Não foi possível converter a resposta '{response}' para float. Usando 0.0.")
            individual_scores.append(0.0)
    
    # Calcular métricas agregadas
    if individual_scores:
        best_match = max(individual_scores)
        avg_score = sum(individual_scores) / len(individual_scores)
        # Overall score privilegia a melhor categoria mas considera a média
        overall_score = 0.7 * best_match + 0.3 * avg_score
    else:
        best_match = avg_score = overall_score = 0.0
    
    return {
        "overall_score": overall_score,
        "individual_scores": individual_scores,
        "best_match": best_match,
        "avg_score": avg_score
    }

def evaluate_combined_metrics(categories_llm, top_categories):
    """
    Avalia usando ambas as métricas: LLM-as-a-Judge e BERT-Score.
    Combina os resultados em um único dicionário.
    """
    # Se categories_llm é uma string, converter para lista
    if isinstance(categories_llm, str):
        categories_llm = [cat.strip() for cat in categories_llm.split(';') if cat.strip()]
    
    # Avaliar com LLM-as-a-Judge
    llm_scores = evaluate_correlation_with_ollama(categories_llm, top_categories)
    
    # Avaliar com BERT-Score
    bert_scores = calculate_bert_score(categories_llm, top_categories)
    
    # Combinar os resultados
    combined = {
        # LLM-as-a-Judge scores
        "llm_overall_score": llm_scores["overall_score"],
        "llm_individual_scores": llm_scores["individual_scores"],
        "llm_best_match": llm_scores["best_match"],
        "llm_avg_score": llm_scores["avg_score"],
        
        # BERT-Score metrics
        "bert_precision": bert_scores["bert_precision"],
        "bert_recall": bert_scores["bert_recall"],
        "bert_f1": bert_scores["bert_f1"],
        "bert_individual_f1": bert_scores["bert_individual_f1"],
        
        # Score combinado (média ponderada entre LLM e BERT F1)
        "combined_score": 0.6 * llm_scores["overall_score"] + 0.4 * bert_scores["bert_f1"]
    }
    
    return combined

# --- Funções de Processamento de Dados ---

def load_and_merge_data():
    """Carrega todos os arquivos de entrada."""
    logging.info("Carregando dados das comunidades...")
    
    with open(COMMUNITIES_FILE, 'r') as f:
        communities_data = json.load(f)
    communities_list = communities_data['leiden']

    with open(NODES_FILE, 'r', encoding='utf-8-sig') as f:
        nodes_data = json.load(f)
    nodes = pd.json_normalize(nodes_data)
    
    logging.info(f"Dados carregados. {len(communities_list)} comunidades encontradas.")
    return communities_list, nodes

def keywords_reference(nodes):
    with open('./run142_raw_communities.json', 'r') as f:
        communities2 = json.load(f)
        communities_list2 = communities2['leiden']
    categories = pd.read_json('./categories.jsonl', lines=True)
    
    categories_merged = categories.merge(
        nodes[["d.identity", "d.properties.title_encode"]],
        left_on="title",
        right_on="d.properties.title_encode",
        how="left"
    )

    categories_merged = categories_merged.drop(columns=["d.properties.title_encode"])
    community_map = {}
    for i, ids in enumerate(communities_list2):
        for node_id in ids:
            community_map[node_id] = i

    categories_merged['community_id'] = categories_merged['d.identity'].map(community_map)

    top20_per_community_wiki = {}

    for i in range(len(communities_list2)):
        subset = categories_merged[categories_merged['community_id'] == i]

        # 'categorias' e 'categorias_ocultas' são listas — precisamos achatar
        all_cats = [cat for cats in subset['categorias'].dropna() for cat in cats]
        all_hidden = [cat for cats in subset['categorias_ocultas'].dropna() for cat in cats]

        top_cats = Counter(all_cats).most_common(20)
        top_hidden = Counter(all_hidden).most_common(20)

        top20_per_community_wiki[i] = {
            "top_categorias": top_cats,
            "top_categorias_ocultas": top_hidden
        }

    return top20_per_community_wiki

def prepare_keywords_df():
    """Prepara o DataFrame com as palavras-chave do algoritmo Leiden."""
    logging.info("Carregando e processando palavras-chave das comunidades...")
    with open(KEYWORDS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rows = []
    # Focar apenas no algoritmo 'leiden'
    leiden_communities = data.get('leiden', {})
    for comm_id, comm_data in leiden_communities.items():
        keywords = comm_data.get('keywords', [])
        keywords_list = [kw[0] for kw in keywords]
        
        rows.append({
            'community_id': int(comm_id),
            'keywords_string': ', '.join(keywords_list),
            'num_keywords': len(keywords_list)
        })

    return pd.DataFrame(rows)

def process_with_llm_in_parallel(df, task_function, input_col_name, output_col_name, extra_args=None):
    """Função genérica para processar um DataFrame em paralelo usando uma função LLM."""
    results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_id = {}
        for idx, row in df.iterrows():
            if extra_args:
                # Para avaliação, que precisa de mais de um argumento
                args = [row[arg] for arg in extra_args]
                future = executor.submit(task_function, row[input_col_name], *args)
            else:
                # Para categorização
                future = executor.submit(task_function, row[input_col_name])
            future_to_id[future] = row['community_id']

        total = len(future_to_id)
        processed = 0
        for future in as_completed(future_to_id):
            comm_id = future_to_id[future]
            try:
                result = future.result()
                results[comm_id] = result
                processed += 1
                if processed % 3 == 0 or processed == total:
                    logging.info(f"Processado {processed}/{total} para '{output_col_name}'.")
            except Exception as exc:
                logging.error(f'Comunidade {comm_id} gerou uma exceção: {exc}')
                results[comm_id] = "erro_processamento"
    
    df[output_col_name] = df['community_id'].map(results)
    return df

# --- Funções de Análise e Geração de Relatório ---

def analyze_results(df_final):
    """Calcula métricas de análise sobre os resultados finais."""
    logging.info("Calculando métricas de análise...")
    
    # Extrair scores dos dicionários - LLM-as-a-Judge
    df_final['llm_overall_score'] = df_final['evaluation_metrics'].apply(
        lambda x: x.get('llm_overall_score', 0.0) if isinstance(x, dict) else 0.0
    )
    df_final['llm_best_match'] = df_final['evaluation_metrics'].apply(
        lambda x: x.get('llm_best_match', 0.0) if isinstance(x, dict) else 0.0
    )
    df_final['llm_avg_score'] = df_final['evaluation_metrics'].apply(
        lambda x: x.get('llm_avg_score', 0.0) if isinstance(x, dict) else 0.0
    )
    
    # Extrair scores do BERT
    df_final['bert_precision'] = df_final['evaluation_metrics'].apply(
        lambda x: x.get('bert_precision', 0.0) if isinstance(x, dict) else 0.0
    )
    df_final['bert_recall'] = df_final['evaluation_metrics'].apply(
        lambda x: x.get('bert_recall', 0.0) if isinstance(x, dict) else 0.0
    )
    df_final['bert_f1'] = df_final['evaluation_metrics'].apply(
        lambda x: x.get('bert_f1', 0.0) if isinstance(x, dict) else 0.0
    )
    
    # Score combinado
    df_final['combined_score'] = df_final['evaluation_metrics'].apply(
        lambda x: x.get('combined_score', 0.0) if isinstance(x, dict) else 0.0
    )
    
    df_final['num_categories'] = df_final['category_llm'].apply(
        lambda x: len(x) if isinstance(x, list) else 1
    )
    
    metrics = {}
    
    # Métricas gerais - LLM-as-a-Judge
    metrics['total_communities'] = len(df_final)
    metrics['mean_llm_overall_score'] = df_final['llm_overall_score'].mean()
    metrics['median_llm_overall_score'] = df_final['llm_overall_score'].median()
    metrics['std_dev_llm_overall'] = df_final['llm_overall_score'].std()
    
    # Métricas gerais - BERT-Score
    metrics['mean_bert_precision'] = df_final['bert_precision'].mean()
    metrics['mean_bert_recall'] = df_final['bert_recall'].mean()
    metrics['mean_bert_f1'] = df_final['bert_f1'].mean()
    metrics['median_bert_f1'] = df_final['bert_f1'].median()
    metrics['std_dev_bert_f1'] = df_final['bert_f1'].std()
    
    # Métricas do score combinado
    metrics['mean_combined_score'] = df_final['combined_score'].mean()
    metrics['median_combined_score'] = df_final['combined_score'].median()
    metrics['std_dev_combined'] = df_final['combined_score'].std()
    
    # Métricas por threshold - LLM
    thresholds = [0.5, 0.6, 0.7, 0.8]
    for threshold in thresholds:
        count_llm = (df_final['llm_overall_score'] >= threshold).sum()
        percentage_llm = (count_llm / metrics['total_communities']) * 100 if metrics['total_communities'] > 0 else 0
        metrics[f'llm_match_count_{int(threshold*100)}'] = count_llm
        metrics[f'llm_match_percentage_{int(threshold*100)}'] = percentage_llm
        
        # Threshold para BERT F1
        count_bert = (df_final['bert_f1'] >= threshold).sum()
        percentage_bert = (count_bert / metrics['total_communities']) * 100 if metrics['total_communities'] > 0 else 0
        metrics[f'bert_match_count_{int(threshold*100)}'] = count_bert
        metrics[f'bert_match_percentage_{int(threshold*100)}'] = percentage_bert
        
        # Threshold para score combinado
        count_combined = (df_final['combined_score'] >= threshold).sum()
        percentage_combined = (count_combined / metrics['total_communities']) * 100 if metrics['total_communities'] > 0 else 0
        metrics[f'combined_match_count_{int(threshold*100)}'] = count_combined
        metrics[f'combined_match_percentage_{int(threshold*100)}'] = percentage_combined
    
    # Métricas de best_match LLM
    metrics['mean_llm_best_match'] = df_final['llm_best_match'].mean()
    metrics['median_llm_best_match'] = df_final['llm_best_match'].median()
    
    # Métricas de número de categorias
    metrics['avg_categories_per_community'] = df_final['num_categories'].mean()
    metrics['communities_with_1_category'] = (df_final['num_categories'] == 1).sum()
    metrics['communities_with_2_categories'] = (df_final['num_categories'] == 2).sum()
    metrics['communities_with_3_categories'] = (df_final['num_categories'] == 3).sum()
    
    # Correlações entre número de categorias e scores
    if df_final['num_categories'].std() > 0:
        metrics['correlation_num_cats_vs_llm_score'] = df_final['num_categories'].corr(df_final['llm_overall_score'])
        metrics['correlation_num_cats_vs_bert_f1'] = df_final['num_categories'].corr(df_final['bert_f1'])
        metrics['correlation_num_cats_vs_combined'] = df_final['num_categories'].corr(df_final['combined_score'])
    else:
        metrics['correlation_num_cats_vs_llm_score'] = 0.0
        metrics['correlation_num_cats_vs_bert_f1'] = 0.0
        metrics['correlation_num_cats_vs_combined'] = 0.0
    
    # Correlação entre LLM e BERT scores
    metrics['correlation_llm_vs_bert'] = df_final['llm_overall_score'].corr(df_final['bert_f1'])
    
    # Identificar melhores e piores (usando combined_score)
    df_sorted = df_final.sort_values(by='combined_score', ascending=False)
    metrics['top_5'] = df_sorted.head(5)
    metrics['bottom_5'] = df_sorted.tail(5)
    
    # Distribuição de scores por quartil
    metrics['llm_quartiles'] = df_final['llm_overall_score'].quantile([0.25, 0.5, 0.75]).to_dict()
    metrics['bert_quartiles'] = df_final['bert_f1'].quantile([0.25, 0.5, 0.75]).to_dict()
    metrics['combined_quartiles'] = df_final['combined_score'].quantile([0.25, 0.5, 0.75]).to_dict()
    
    return metrics

def generate_markdown_report(df_final, metrics, top_wiki_categories):
    """Gera um arquivo de relatório em formato Markdown."""
    logging.info(f"Gerando relatório em '{REPORT_FILE}'...")
    
    # Helper function para formatar as top keywords de referência
    def format_top_cats(comm_id):
        cats = top_wiki_categories.get(comm_id, {}).get('top_categorias', [])
        if not cats:
            return "N/A"
        return ", ".join([f"'{cat}'" for cat, _ in cats[:5]])
    
    # Helper function para formatar categorias LLM
    def format_llm_categories(cats):
        if isinstance(cats, list):
            return " / ".join(cats)  # Usar "/" ao invés de "|" para não quebrar colunas Markdown
        return str(cats)
    
    # Helper function para formatar scores individuais LLM
    def format_llm_scores(eval_dict):
        if isinstance(eval_dict, dict) and 'llm_individual_scores' in eval_dict:
            scores = eval_dict['llm_individual_scores']
            return ", ".join([f"{s:.3f}" for s in scores])
        return "N/A"
    
    # Helper function para formatar scores individuais BERT
    def format_bert_scores(eval_dict):
        if isinstance(eval_dict, dict) and 'bert_individual_f1' in eval_dict:
            scores = eval_dict['bert_individual_f1']
            return ", ".join([f"{s:.3f}" for s in scores])
        return "N/A"

    report = []
    report.append(f"# Análise de Nomenclatura de Comunidades (Leiden)")
    report.append(f"## Avaliação Multi-Métrica: LLM-as-a-Judge + BERT-Score")
    report.append(f"Relatório gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Modelo LLM: {OLLAMA_MODEL}")
    report.append(f"Modelo BERT: {BERT_MODEL}")
    report.append(f"\n**Nota**: As avaliações comparam as categorias geradas pelo LLM com as keywords extraídas das comunidades.")
    report.append("\n---")
    
    report.append("## Resumo Executivo das Métricas")
    report.append(f"- **Total de Comunidades Analisadas**: {metrics['total_communities']}")
    report.append(f"- **Média de Categorias por Comunidade**: {metrics['avg_categories_per_community']:.2f}")
    report.append(f"  - Comunidades com 1 categoria: {metrics['communities_with_1_category']}")
    report.append(f"  - Comunidades com 2 categorias: {metrics['communities_with_2_categories']}")
    report.append(f"  - Comunidades com 3 categorias: {metrics['communities_with_3_categories']}")
    
    report.append("\n### 📊 Métricas de Score Combinado (60% LLM + 40% BERT)")
    report.append(f"- **Score Médio Combinado**: {metrics['mean_combined_score']:.3f}")
    report.append(f"- **Score Mediano Combinado**: {metrics['median_combined_score']:.3f}")
    report.append(f"- **Desvio Padrão**: {metrics['std_dev_combined']:.3f}")
    
    report.append("\n### 🤖 Métricas LLM-as-a-Judge")
    report.append(f"- **Score Médio LLM**: {metrics['mean_llm_overall_score']:.3f}")
    report.append(f"- **Score Mediano LLM**: {metrics['median_llm_overall_score']:.3f}")
    report.append(f"- **Desvio Padrão LLM**: {metrics['std_dev_llm_overall']:.3f}")
    report.append(f"- **Média do Melhor Match LLM**: {metrics['mean_llm_best_match']:.3f}")
    
    report.append("\n### 📝 Métricas BERT-Score")
    report.append(f"- **Precision Média**: {metrics['mean_bert_precision']:.3f}")
    report.append(f"- **Recall Médio**: {metrics['mean_bert_recall']:.3f}")
    report.append(f"- **F1 Médio**: {metrics['mean_bert_f1']:.3f}")
    report.append(f"- **F1 Mediano**: {metrics['median_bert_f1']:.3f}")
    report.append(f"- **Desvio Padrão F1**: {metrics['std_dev_bert_f1']:.3f}")
    
    report.append("\n### Distribuição por Threshold - Score Combinado")
    report.append(f"- **Score >= 0.8**: {metrics['combined_match_count_80']} comunidades ({metrics['combined_match_percentage_80']:.2f}%)")
    report.append(f"- **Score >= 0.7**: {metrics['combined_match_count_70']} comunidades ({metrics['combined_match_percentage_70']:.2f}%)")
    report.append(f"- **Score >= 0.6**: {metrics['combined_match_count_60']} comunidades ({metrics['combined_match_percentage_60']:.2f}%)")
    report.append(f"- **Score >= 0.5**: {metrics['combined_match_count_50']} comunidades ({metrics['combined_match_percentage_50']:.2f}%)")
    
    report.append("\n### Distribuição por Threshold - LLM")
    report.append(f"- **Score >= 0.7**: {metrics['llm_match_count_70']} comunidades ({metrics['llm_match_percentage_70']:.2f}%)")
    report.append(f"- **Score >= 0.6**: {metrics['llm_match_count_60']} comunidades ({metrics['llm_match_percentage_60']:.2f}%)")
    
    report.append("\n### Distribuição por Threshold - BERT F1")
    report.append(f"- **F1 >= 0.7**: {metrics['bert_match_count_70']} comunidades ({metrics['bert_match_percentage_70']:.2f}%)")
    report.append(f"- **F1 >= 0.6**: {metrics['bert_match_count_60']} comunidades ({metrics['bert_match_percentage_60']:.2f}%)")
    
    report.append("\n### Correlações")
    report.append(f"- **Correlação LLM vs BERT**: {metrics['correlation_llm_vs_bert']:.3f}")
    report.append(f"- **Correlação Nº Categorias vs Score Combinado**: {metrics['correlation_num_cats_vs_combined']:.3f}")
    report.append(f"- **Correlação Nº Categorias vs LLM**: {metrics['correlation_num_cats_vs_llm_score']:.3f}")
    report.append(f"- **Correlação Nº Categorias vs BERT**: {metrics['correlation_num_cats_vs_bert_f1']:.3f}")
    
    report.append("\n### Quartis de Distribuição")
    report.append("#### Score Combinado:")
    report.append(f"- **25º Percentil**: {metrics['combined_quartiles'][0.25]:.3f}")
    report.append(f"- **50º Percentil (Mediana)**: {metrics['combined_quartiles'][0.50]:.3f}")
    report.append(f"- **75º Percentil**: {metrics['combined_quartiles'][0.75]:.3f}")
    report.append("\n#### LLM Score:")
    report.append(f"- **25º Percentil**: {metrics['llm_quartiles'][0.25]:.3f}")
    report.append(f"- **50º Percentil (Mediana)**: {metrics['llm_quartiles'][0.50]:.3f}")
    report.append(f"- **75º Percentil**: {metrics['llm_quartiles'][0.75]:.3f}")
    report.append("\n#### BERT F1:")
    report.append(f"- **25º Percentil**: {metrics['bert_quartiles'][0.25]:.3f}")
    report.append(f"- **50º Percentil (Mediana)**: {metrics['bert_quartiles'][0.50]:.3f}")
    report.append(f"- **75º Percentil**: {metrics['bert_quartiles'][0.75]:.3f}")
    report.append("\n---")

    report.append("## Top 5 Melhores Comunidades (Maior Score Combinado)")
    for _, row in metrics['top_5'].iterrows():
        report.append(f"### Comunidade {row['community_id']}")
        report.append(f"**Score Combinado**: {row['combined_score']:.3f} | **LLM**: {row['llm_overall_score']:.3f} | **BERT F1**: {row['bert_f1']:.3f}")
        report.append(f"- **Categorias Geradas (LLM)**: `{format_llm_categories(row['category_llm'])}`")
        report.append(f"- **Número de Categorias**: {row['num_categories']}")
        report.append(f"- **LLM Scores** (Best: {row['llm_best_match']:.3f}, Avg: {row['llm_avg_score']:.3f}): {format_llm_scores(row['evaluation_metrics'])}")
        report.append(f"- **BERT Scores** (P: {row['bert_precision']:.3f}, R: {row['bert_recall']:.3f}, F1s): {format_bert_scores(row['evaluation_metrics'])}")
        report.append(f"- **Palavras-chave**: _{row['keywords_string']}_")
        report.append(f"- **Top 5 Categorias de Referência**: {format_top_cats(row['community_id'])}")
        report.append("")
    report.append("\n---")
    
    report.append("## Top 5 Piores Comunidades (Menor Score Combinado)")
    for _, row in metrics['bottom_5'].iterrows():
        report.append(f"### Comunidade {row['community_id']}")
        report.append(f"**Score Combinado**: {row['combined_score']:.3f} | **LLM**: {row['llm_overall_score']:.3f} | **BERT F1**: {row['bert_f1']:.3f}")
        report.append(f"- **Categorias Geradas (LLM)**: `{format_llm_categories(row['category_llm'])}`")
        report.append(f"- **Número de Categorias**: {row['num_categories']}")
        report.append(f"- **LLM Scores** (Best: {row['llm_best_match']:.3f}, Avg: {row['llm_avg_score']:.3f}): {format_llm_scores(row['evaluation_metrics'])}")
        report.append(f"- **BERT Scores** (P: {row['bert_precision']:.3f}, R: {row['bert_recall']:.3f}, F1s): {format_bert_scores(row['evaluation_metrics'])}")
        report.append(f"- **Palavras-chave**: _{row['keywords_string']}_")
        report.append(f"- **Top 5 Categorias de Referência**: {format_top_cats(row['community_id'])}")
        report.append("")
    report.append("\n---")
    
    report.append("## Análise de Distribuição por Número de Categorias")
    for num_cats in [1, 2, 3]:
        subset = df_final[df_final['num_categories'] == num_cats]
        if len(subset) > 0:
            report.append(f"\n### Comunidades com {num_cats} Categoria(s) ({len(subset)} total)")
            report.append(f"- **Score Médio Combinado**: {subset['combined_score'].mean():.3f}")
            report.append(f"- **Score Médio LLM**: {subset['llm_overall_score'].mean():.3f}")
            report.append(f"- **Score Médio BERT F1**: {subset['bert_f1'].mean():.3f}")
            report.append(f"- **Melhor Score Combinado**: {subset['combined_score'].max():.3f}")
            report.append(f"- **Pior Score Combinado**: {subset['combined_score'].min():.3f}")
    report.append("\n---")
    
    report.append("## Tabela Completa de Resultados")
    # Preparar DataFrame para exibição
    df_display = df_final.copy()
    df_display['categories_llm_display'] = df_display['category_llm'].apply(format_llm_categories)
    df_display['top_5_keywords'] = df_display['community_id'].apply(format_top_cats)
    df_display['llm_scores_display'] = df_display['evaluation_metrics'].apply(format_llm_scores)
    df_display['bert_scores_display'] = df_display['evaluation_metrics'].apply(format_bert_scores)
    
    # Criar tabela manualmente para evitar problemas de formatação
    report.append("\n| ID | Categorias | # Cat | Score Comb | LLM Score | BERT F1 | BERT Prec | BERT Rec | LLM Scores | BERT F1s | Top 5 Categorias |")
    report.append("|---:|:-----------|------:|-----------:|----------:|--------:|----------:|---------:|:-----------|:---------|:---------------|")
    
    for _, row in df_display.iterrows():
        report.append(
            f"| {int(row['community_id'])} "
            f"| {format_llm_categories(row['category_llm'])} "
            f"| {int(row['num_categories'])} "
            f"| {float(row['combined_score']):.3f} "
            f"| {float(row['llm_overall_score']):.3f} "
            f"| {float(row['bert_f1']):.3f} "
            f"| {float(row['bert_precision']):.3f} "
            f"| {float(row['bert_recall']):.3f} "
            f"| {row['llm_scores_display']} "
            f"| {row['bert_scores_display']} "
            f"| {format_top_cats(row['community_id'])} |"
        )

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
    logging.info("Relatório salvo com sucesso.")

# --- Execução Principal ---

def main():
    # Passo 1: Carregar dados
    communities_list, nodes = load_and_merge_data()
    
    # Passo 2: Preparar DataFrame de palavras-chave
    df_keywords = prepare_keywords_df()
    
    # Passo 3: Criar estrutura de referência usando o arquivo categories.jsonl
    top_keywords_reference = keywords_reference(nodes)
    
    # Passo 4: Nomear comunidades com LLM
    logging.info("Iniciando a categorização de comunidades com Ollama...")
    df_categorized = process_with_llm_in_parallel(
        df_keywords, 
        task_function=categorize_with_ollama, 
        input_col_name='keywords_string', 
        output_col_name='category_llm'
    )

    # Adicionar as keywords de referência ao DataFrame para a próxima etapa
    df_categorized['top_wiki_categories'] = df_categorized['community_id'].apply(
        lambda cid: top_keywords_reference.get(cid, {}).get('top_categorias', [])
    )

    # Passo 5: Avaliar com métricas combinadas (LLM + BERT)
    logging.info("Iniciando a avaliação com LLM-as-a-Judge + BERT-Score...")
    df_final = process_with_llm_in_parallel(
        df_categorized, 
        task_function=evaluate_combined_metrics, 
        input_col_name='category_llm', 
        output_col_name='evaluation_metrics',
        extra_args=['top_wiki_categories']
    )

    # Passo 6: Análise dos resultados
    metrics = analyze_results(df_final)
    
    # Passo 7: Geração do relatório
    generate_markdown_report(df_final, metrics, top_keywords_reference)
    
    logging.info("Processo concluído.")
    print("\n=== Resumo de Resultados ===")
    print(f"Total de Comunidades: {metrics['total_communities']}")
    print(f"\n📊 SCORE COMBINADO (60% LLM + 40% BERT):")
    print(f"  - Média: {metrics['mean_combined_score']:.3f}")
    print(f"  - Mediana: {metrics['median_combined_score']:.3f}")
    print(f"  - Score >= 0.7: {metrics['combined_match_count_70']} ({metrics['combined_match_percentage_70']:.2f}%)")
    print(f"\n🤖 LLM-AS-A-JUDGE:")
    print(f"  - Média: {metrics['mean_llm_overall_score']:.3f}")
    print(f"  - Mediana: {metrics['median_llm_overall_score']:.3f}")
    print(f"  - Score >= 0.7: {metrics['llm_match_count_70']} ({metrics['llm_match_percentage_70']:.2f}%)")
    print(f"\n📝 BERT-SCORE:")
    print(f"  - F1 Médio: {metrics['mean_bert_f1']:.3f}")
    print(f"  - Precision Média: {metrics['mean_bert_precision']:.3f}")
    print(f"  - Recall Médio: {metrics['mean_bert_recall']:.3f}")
    print(f"  - F1 >= 0.7: {metrics['bert_match_count_70']} ({metrics['bert_match_percentage_70']:.2f}%)")
    print(f"\n📈 CORRELAÇÕES:")
    print(f"  - LLM vs BERT: {metrics['correlation_llm_vs_bert']:.3f}")
    print(f"\nMédia de Categorias por Comunidade: {metrics['avg_categories_per_community']:.2f}")
    print(f"\nRelatório completo salvo em: {REPORT_FILE}")
    # Salvar também um JSON com resultados detalhados
    output_json = REPORT_FILE.replace('.md', '_detailed.json')
    df_output = df_final.copy()
    df_output['category_llm_str'] = df_output['category_llm'].apply(
        lambda x: x if isinstance(x, list) else [str(x)]
    )
    
    # Converter tipos do pandas para tipos Python nativos para serialização JSON
    def convert_to_native_types(obj):
        """Converte tipos do pandas/numpy para tipos nativos do Python."""
        if isinstance(obj, (np.integer, pd.Int64Dtype)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.to_list()
        elif isinstance(obj, dict):
            return {k: convert_to_native_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native_types(item) for item in obj]
        return obj
    
    # Preparar métricas para JSON
    metrics_for_json = {}
    for k, v in metrics.items():
        if k not in ['top_5', 'bottom_5', 'llm_quartiles', 'bert_quartiles', 'combined_quartiles']:
            metrics_for_json[k] = convert_to_native_types(v)
    
    # Preparar comunidades para JSON
    communities_list = []
    for _, row in df_output.iterrows():
        eval_metrics = row['evaluation_metrics'] if isinstance(row['evaluation_metrics'], dict) else {}
        community_dict = {
            'community_id': int(row['community_id']),
            'categories': row['category_llm_str'],
            'num_categories': int(row['num_categories']),
            'combined_score': float(row['combined_score']),
            'llm_overall_score': float(row['llm_overall_score']),
            'llm_best_match': float(row['llm_best_match']),
            'llm_avg_score': float(row['llm_avg_score']),
            'bert_precision': float(row['bert_precision']),
            'bert_recall': float(row['bert_recall']),
            'bert_f1': float(row['bert_f1']),
            'keywords': row['keywords_string']
        }
        communities_list.append(community_dict)
    
    results_dict = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'total_communities': int(metrics['total_communities']),
            'llm_model': OLLAMA_MODEL,
            'bert_model': BERT_MODEL,
            'evaluation_methods': ['LLM-as-a-Judge', 'BERT-Score'],
            'combined_score_formula': '0.6 * llm_score + 0.4 * bert_f1'
        },
        'metrics': metrics_for_json,
        'communities': communities_list
    }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    logging.info(f"Resultados detalhados salvos em: {output_json}")

if __name__ == "__main__":
    main()