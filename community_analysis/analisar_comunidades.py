import json
import pandas as pd
import numpy as np
import requests
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import logging

# --- Configuração ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Arquivos de entrada
COMMUNITIES_FILE = 'run142_raw_communities.json'
CATEGORIES_FILE = 'categories.jsonl'
NODES_FILE = 'nodes.json'
KEYWORDS_FILE = 'run142_community_keywords.json'

# Arquivo de saída
REPORT_FILE = 'analise_comunidades_leiden.md'

# Configurações do Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
# OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_MODEL = "mistral"
OLLAMA_TIMEOUT = 300 # 5 minutos por request
MAX_WORKERS = 1 # Ajuste conforme a capacidade da sua máquina

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

def evaluate_correlation_with_ollama(categories_llm, top_categories):
    """Avalia a correlação semântica e retorna múltiplos scores."""
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
You are an accurate semantic evaluator. Evaluate the relevance of the following category created by an LLM model relative to a community's top Wikipedia categories.

LLM-generated category: "{cat_llm}"
Top Wikipedia categories: "{top_cats_str}"

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

# --- Funções de Processamento de Dados ---

def load_and_merge_data():
    """Carrega todos os arquivos de entrada e os une em um DataFrame."""
    logging.info("Carregando e unindo dados...")
    
    with open(COMMUNITIES_FILE, 'r') as f:
        communities_data = json.load(f)
    communities_list = communities_data['leiden']

    categories = pd.read_json(CATEGORIES_FILE, lines=True)

    with open(NODES_FILE, 'r', encoding='utf-8-sig') as f:
        nodes_data = json.load(f)
    nodes = pd.json_normalize(nodes_data)

    categories_merged = categories.merge(
        nodes[["d.identity", "d.properties.title_encode"]],
        left_on="title",
        right_on="d.properties.title_encode",
        how="left"
    )
    categories_merged = categories_merged.drop(columns=["d.properties.title_encode"])

    community_map = {node_id: i for i, ids in enumerate(communities_list) for node_id in ids}
    categories_merged['community_id'] = categories_merged['d.identity'].map(community_map)
    
    logging.info(f"Dados carregados. {categories_merged['community_id'].nunique()} comunidades encontradas.")
    return communities_list, categories_merged

def get_top_wiki_categories(communities_list, categories_merged):
    """Extrai as top 20 categorias da Wikipedia para cada comunidade."""
    logging.info("Extraindo top 20 categorias da Wikipedia por comunidade...")
    top20_per_community_wiki = {}
    for i in range(len(communities_list)):
        subset = categories_merged[categories_merged['community_id'] == i]
        
        all_cats = [cat for cats_list in subset['categorias'].dropna() for cat in cats_list]
        top_cats = Counter(all_cats).most_common(20)
        
        top20_per_community_wiki[i] = {"top_categorias": top_cats}
        
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
    
    # Extrair scores dos dicionários
    df_final['overall_score'] = df_final['llm_correlation'].apply(
        lambda x: x.get('overall_score', 0.0) if isinstance(x, dict) else 0.0
    )
    df_final['best_match'] = df_final['llm_correlation'].apply(
        lambda x: x.get('best_match', 0.0) if isinstance(x, dict) else 0.0
    )
    df_final['avg_score'] = df_final['llm_correlation'].apply(
        lambda x: x.get('avg_score', 0.0) if isinstance(x, dict) else 0.0
    )
    df_final['num_categories'] = df_final['category_llm'].apply(
        lambda x: len(x) if isinstance(x, list) else 1
    )
    
    metrics = {}
    
    # Métricas gerais
    metrics['total_communities'] = len(df_final)
    metrics['mean_overall_score'] = df_final['overall_score'].mean()
    metrics['median_overall_score'] = df_final['overall_score'].median()
    metrics['std_dev_overall'] = df_final['overall_score'].std()
    
    # Métricas por threshold
    thresholds = [0.5, 0.6, 0.7, 0.8]
    for threshold in thresholds:
        count = (df_final['overall_score'] >= threshold).sum()
        percentage = (count / metrics['total_communities']) * 100 if metrics['total_communities'] > 0 else 0
        metrics[f'match_count_{int(threshold*100)}'] = count
        metrics[f'match_percentage_{int(threshold*100)}'] = percentage
    
    # Métricas de best_match
    metrics['mean_best_match'] = df_final['best_match'].mean()
    metrics['median_best_match'] = df_final['best_match'].median()
    
    # Métricas de número de categorias
    metrics['avg_categories_per_community'] = df_final['num_categories'].mean()
    metrics['communities_with_1_category'] = (df_final['num_categories'] == 1).sum()
    metrics['communities_with_2_categories'] = (df_final['num_categories'] == 2).sum()
    metrics['communities_with_3_categories'] = (df_final['num_categories'] == 3).sum()
    
    # Correlação entre número de categorias e score
    if df_final['num_categories'].std() > 0:
        metrics['correlation_num_cats_vs_score'] = df_final['num_categories'].corr(df_final['overall_score'])
    else:
        metrics['correlation_num_cats_vs_score'] = 0.0
    
    # Identificar melhores e piores (usando overall_score)
    df_sorted = df_final.sort_values(by='overall_score', ascending=False)
    metrics['top_5'] = df_sorted.head(5)
    metrics['bottom_5'] = df_sorted.tail(5)
    
    # Distribuição de scores por quartil
    metrics['quartiles'] = df_final['overall_score'].quantile([0.25, 0.5, 0.75]).to_dict()
    
    return metrics

def generate_markdown_report(df_final, metrics, top_wiki_categories):
    """Gera um arquivo de relatório em formato Markdown."""
    logging.info(f"Gerando relatório em '{REPORT_FILE}'...")
    
    # Helper function para formatar as top categorias
    def format_top_cats(comm_id):
        cats = top_wiki_categories.get(comm_id, {}).get('top_categorias', [])
        if not cats:
            return "N/A"
        return ", ".join([f"'{cat}' ({count})" for cat, count in cats[:5]])
    
    # Helper function para formatar categorias LLM
    def format_llm_categories(cats):
        if isinstance(cats, list):
            return " | ".join(cats)
        return str(cats)
    
    # Helper function para formatar scores individuais
    def format_individual_scores(corr_dict):
        if isinstance(corr_dict, dict) and 'individual_scores' in corr_dict:
            scores = corr_dict['individual_scores']
            return ", ".join([f"{s:.3f}" for s in scores])
        return "N/A"

    report = []
    report.append(f"# Análise de Nomenclatura de Comunidades (Leiden)")
    report.append(f"Relatório gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n---")
    
    report.append("## Resumo Executivo das Métricas")
    report.append(f"- **Total de Comunidades Analisadas**: {metrics['total_communities']}")
    report.append(f"- **Média de Categorias por Comunidade**: {metrics['avg_categories_per_community']:.2f}")
    report.append(f"  - Comunidades com 1 categoria: {metrics['communities_with_1_category']}")
    report.append(f"  - Comunidades com 2 categorias: {metrics['communities_with_2_categories']}")
    report.append(f"  - Comunidades com 3 categorias: {metrics['communities_with_3_categories']}")
    report.append("\n### Métricas de Score Overall (combinação ponderada)")
    report.append(f"- **Score Médio Overall**: {metrics['mean_overall_score']:.3f}")
    report.append(f"- **Score Mediano Overall**: {metrics['median_overall_score']:.3f}")
    report.append(f"- **Desvio Padrão**: {metrics['std_dev_overall']:.3f}")
    report.append("\n### Distribuição por Threshold")
    report.append(f"- **Score >= 0.8**: {metrics['match_count_80']} comunidades ({metrics['match_percentage_80']:.2f}%)")
    report.append(f"- **Score >= 0.7**: {metrics['match_count_70']} comunidades ({metrics['match_percentage_70']:.2f}%)")
    report.append(f"- **Score >= 0.6**: {metrics['match_count_60']} comunidades ({metrics['match_percentage_60']:.2f}%)")
    report.append(f"- **Score >= 0.5**: {metrics['match_count_50']} comunidades ({metrics['match_percentage_50']:.2f}%)")
    report.append("\n### Métricas de Best Match")
    report.append(f"- **Média do Melhor Match**: {metrics['mean_best_match']:.3f}")
    report.append(f"- **Mediana do Melhor Match**: {metrics['median_best_match']:.3f}")
    report.append("\n### Correlação")
    report.append(f"- **Correlação entre Nº de Categorias e Score**: {metrics['correlation_num_cats_vs_score']:.3f}")
    report.append("\n### Quartis de Distribuição")
    report.append(f"- **25º Percentil**: {metrics['quartiles'][0.25]:.3f}")
    report.append(f"- **50º Percentil (Mediana)**: {metrics['quartiles'][0.50]:.3f}")
    report.append(f"- **75º Percentil**: {metrics['quartiles'][0.75]:.3f}")
    report.append("\n---")

    report.append("## Top 5 Melhores Comunidades (Maior Correlação)")
    for _, row in metrics['top_5'].iterrows():
        report.append(f"### Comunidade {row['community_id']} (Overall: {row['overall_score']:.3f} | Best: {row['best_match']:.3f} | Avg: {row['avg_score']:.3f})")
        report.append(f"- **Categorias Geradas (LLM)**: `{format_llm_categories(row['category_llm'])}`")
        report.append(f"- **Número de Categorias**: {row['num_categories']}")
        report.append(f"- **Scores Individuais**: {format_individual_scores(row['llm_correlation'])}")
        report.append(f"- **Palavras-chave**: _{row['keywords_string']}_")
        report.append(f"- **Top 5 Categorias Wikipedia**: {format_top_cats(row['community_id'])}")
        report.append("")
    report.append("\n---")
    
    report.append("## Top 5 Piores Comunidades (Menor Correlação)")
    for _, row in metrics['bottom_5'].iterrows():
        report.append(f"### Comunidade {row['community_id']} (Overall: {row['overall_score']:.3f} | Best: {row['best_match']:.3f} | Avg: {row['avg_score']:.3f})")
        report.append(f"- **Categorias Geradas (LLM)**: `{format_llm_categories(row['category_llm'])}`")
        report.append(f"- **Número de Categorias**: {row['num_categories']}")
        report.append(f"- **Scores Individuais**: {format_individual_scores(row['llm_correlation'])}")
        report.append(f"- **Palavras-chave**: _{row['keywords_string']}_")
        report.append(f"- **Top 5 Categorias Wikipedia**: {format_top_cats(row['community_id'])}")
        report.append("")
    report.append("\n---")
    
    report.append("## Análise de Distribuição por Número de Categorias")
    for num_cats in [1, 2, 3]:
        subset = df_final[df_final['num_categories'] == num_cats]
        if len(subset) > 0:
            report.append(f"\n### Comunidades com {num_cats} Categoria(s) ({len(subset)} total)")
            report.append(f"- **Score Médio Overall**: {subset['overall_score'].mean():.3f}")
            report.append(f"- **Score Mediano Overall**: {subset['overall_score'].median():.3f}")
            report.append(f"- **Melhor Score**: {subset['overall_score'].max():.3f}")
            report.append(f"- **Pior Score**: {subset['overall_score'].min():.3f}")
    report.append("\n---")
    
    report.append("## Tabela Completa de Resultados")
    # Preparar DataFrame para exibição
    df_display = df_final.copy()
    df_display['categories_llm_display'] = df_display['category_llm'].apply(format_llm_categories)
    df_display['top_5_wiki_categories'] = df_display['community_id'].apply(format_top_cats)
    df_display['individual_scores_display'] = df_display['llm_correlation'].apply(format_individual_scores)
    
    report.append(df_display[['community_id', 'categories_llm_display', 'num_categories', 
                               'overall_score', 'best_match', 'avg_score', 
                               'individual_scores_display', 'top_5_wiki_categories']].to_markdown(index=False))

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
    logging.info("Relatório salvo com sucesso.")

# --- Execução Principal ---

def main():
    # Passo 1 e 2: Carregar dados e extrair categorias
    communities_list, categories_merged = load_and_merge_data()
    top_wiki_categories = get_top_wiki_categories(communities_list, categories_merged)
    
    # Passo 3: Preparar DataFrame de palavras-chave
    df_keywords = prepare_keywords_df()
    
    # Passo 4: Nomear comunidades com LLM
    logging.info("Iniciando a categorização de comunidades com Ollama...")
    df_categorized = process_with_llm_in_parallel(
        df_keywords, 
        task_function=categorize_with_ollama, 
        input_col_name='keywords_string', 
        output_col_name='category_llm'
    )

    # Adicionar as categorias da Wikipedia ao DataFrame para a próxima etapa
    df_categorized['top_wiki_categories'] = df_categorized['community_id'].apply(
        lambda cid: top_wiki_categories.get(cid, {}).get('top_categorias', [])
    )

    # Passo 5: Avaliar a correlação com LLM
    logging.info("Iniciando a avaliação da correlação com Ollama...")
    df_final = process_with_llm_in_parallel(
        df_categorized, 
        task_function=evaluate_correlation_with_ollama, 
        input_col_name='category_llm', 
        output_col_name='llm_correlation',
        extra_args=['top_wiki_categories']
    )

    # Passo 6: Análise dos resultados
    metrics = analyze_results(df_final)    # Passo 7: Geração do relatório
    generate_markdown_report(df_final, metrics, top_wiki_categories)
    
    logging.info("Processo concluído.")
    print("\n=== Resumo de Resultados ===")
    print(f"Total de Comunidades: {metrics['total_communities']}")
    print(f"Score Médio Overall: {metrics['mean_overall_score']:.3f}")
    print(f"Score Mediano Overall: {metrics['median_overall_score']:.3f}")
    print(f"Média de Categorias por Comunidade: {metrics['avg_categories_per_community']:.2f}")
    print(f"Comunidades com Score >= 0.7: {metrics['match_count_70']} ({metrics['match_percentage_70']:.2f}%)")
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
        import numpy as np
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
        if k not in ['top_5', 'bottom_5', 'quartiles']:
            metrics_for_json[k] = convert_to_native_types(v)
    
    # Preparar comunidades para JSON
    communities_list = []
    for _, row in df_output.iterrows():
        community_dict = {
            'community_id': int(row['community_id']),
            'categories': row['category_llm_str'],
            'num_categories': int(row['num_categories']),
            'overall_score': float(row['overall_score']),
            'best_match': float(row['best_match']),
            'avg_score': float(row['avg_score']),
            'keywords': row['keywords_string']
        }
        communities_list.append(community_dict)
    
    results_dict = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'total_communities': int(metrics['total_communities']),
            'model_used': OLLAMA_MODEL
        },
        'metrics': metrics_for_json,
        'communities': communities_list
    }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    logging.info(f"Resultados detalhados salvos em: {output_json}")

if __name__ == "__main__":
    main()