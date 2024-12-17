from datetime import datetime, date
import json
import requests
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd
import random


def generate_github_URL(base_url, endpoint, filter_params=None):

    # Crea la consulta de filtros por columna si hay filtros
    if filter_params:
        filter_query = '+'.join([f"{key}:{value}" for key, value in filter_params.items()])
        return f"{base_url}{endpoint}?q={filter_query}"
    
    # Si no hay filtros, solo crea la URL con q= vacío
    return f"{base_url}/{endpoint}?q="


def get_repositories_by_year(headers, date_range, columns):
    all_repositories_by_year = {} # Diccionário para almacenar todos los repositorios de la consulta

    for year in date_range:
        all_repositories = []  # Lista para almacenar todos los repositorios del año
        print(f"Iniciando la recolección de datos para el año {year}...")

        for month in range(1, 13):  # De enero a diciembre
            start_date = date(year, month, 1)

            # Define el inicio del próximo año cuando es diciembre
            if month == 12:
                end_date = date(year + 1, 1, 1)
            else:
                end_date = date(year, month + 1, 1)
            
            # Crea la url con las fechas de inicio y fim del més
            url = f'https://api.github.com/search/repositories?q=topic:data-science+created:{start_date}..{end_date}'

            # Definición de page = 1 para iniciar el conteo
            page = 1

            while True:
                time.sleep(3) # Espera un poco antes de intentar nuevamente para no sobrecargar  GitHub
                paginated_url = f"{url}&page={page}&per_page=100"
                print(f"\nConsultando página {page}: {start_date} - {end_date}")
                response = requests.get(paginated_url, headers=headers)

                # Verifica si la conexión está bién
                if response.status_code != 200:
                    print(f"Error al acceder a la página {page} para {start_date}: {response.status_code}")
                    time.sleep(3)  # Espera un poco antes de intentar nuevamente para no sobrecargar  GitHub
                    continue

                data = response.json()
                items = data.get('items', [])

                if not items:
                    print(f"No hay elementos en la página {page} para {start_date} - {end_date}, pasando al próximo rango de fechas")
                    break

                # Filtra los elementos según las columnas especificadas
                filtered_items = filter_repository_data(items, columns)
                all_repositories.extend(filtered_items)
                print(f"Página {page} de {start_date} - {end_date} guardada con éxito")

                # Respeta el límite de la API
                time.sleep(3) # Espera un poco antes de intentar nuevamente para no sobrecargar  GitHub
                page += 1
        
        # Almacena los repositorios del año
        all_repositories_by_year[year] = all_repositories
        print(f"Datos de {year} recolectados con éxito")

        # Verifica si se recolectaron repositorios
        if all_repositories:
            filename = f'repositories_{year}.json'
            with open(filename, 'w', encoding='utf-8') as file:
                json.dump(all_repositories, file, ensure_ascii=False, indent=4)
            print(f"Datos de {year} guardados en '{filename}'.")
        else:
            print(f"No se encontraron repositorios para el año {year}. No se guardó ningún archivo.")


def filter_repository_data(items, columns):
    filtered_items = []
    for item in items:
        filtered_item = {col: item.get(col) for col in columns}
        filtered_items.append(filtered_item)
    return filtered_items

def contar_datos_por_agno(df, column_datetime, color='cornflowerblue'):
    
    year_counts = df['year'].value_counts().sort_index()

    plt.figure(figsize=(12, 7))
    bars = plt.bar(year_counts.index, year_counts.values, color=color)

    plt.title('Número de Datos por Año (2010-2024)', fontsize=16, pad=20)
    plt.xlabel('Año', fontsize=14, labelpad=15)
    plt.ylabel('Conteo', fontsize=14, labelpad=15)

    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), 
                 f'{bar.get_height()}', ha='center', va='bottom', fontsize=12)

    plt.xticks(year_counts.index, rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

def generate_random_color():
    r = random.randint(0, 180)
    g = random.randint(0, 180)
    b = random.randint(0, 200)
    
    return "#{:02x}{:02x}{:02x}".format(r, g, b)

def pinta_distribucion_categoricas(df, columnas_categoricas, relativa=False, mostrar_valores=False):
    num_columnas = len(columnas_categoricas)
    num_filas = (num_columnas // 2) + (num_columnas % 2)

    fig, axes = plt.subplots(num_filas, 2, figsize=(15, 5 * num_filas))
    axes = axes.flatten() 

    for i, col in enumerate(columnas_categoricas):
        ax = axes[i]
        if relativa:
            total = df[col].value_counts().sum()
            serie = df[col].value_counts().apply(lambda x: x / total)
            sns.barplot(x=serie.index, y=serie, ax=ax, palette='viridis')  # Removido o hue
            ax.set_ylabel('Frecuencia Relativa')
        else:
            serie = df[col].value_counts()
            sns.barplot(x=serie.index, y=serie, ax=ax, palette='viridis')  # Removido o hue
            ax.set_ylabel('Frecuencia')

        ax.set_title(f'Distribución de {col}')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)

        if mostrar_valores:
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height), 
                            ha='center', va='center', xytext=(0, 9), textcoords='offset points')

    for j in range(i + 1, num_filas * 2):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def plot_top_20_numerical(df, sorted_value_counts, color='royalblue'):

    top_20_items = dict(list(sorted_value_counts.items())[:20])

    max_count = max(top_20_items.values())

    plt.figure(figsize=(12, 7))
    bars = plt.bar(top_20_items.keys(), top_20_items.values(), color=color)

    plt.title(f'Top 20 Valores Más Frecuentes', fontsize=16)
    plt.xlabel('Valores', fontsize=14, labelpad=20)
    plt.ylabel('Frecuencia', fontsize=14, labelpad=20)

    for bar in bars:
        yval = bar.get_height()
        normalized_percentage = (yval / max_count) * 100
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{normalized_percentage:.1f}%', 
                 ha='center', va='bottom', fontsize=12)

    plt.xticks(rotation=45, ha='right', fontsize=12)

    plt.subplots_adjust(bottom=0.25, top=0.9)

    plt.tight_layout()

    plt.show()

def count_topics(df, column_name):
    all_topics = [topic for sublist in df[column_name] for topic in sublist]
    
    topic_counts = Counter(all_topics)
    
    sorted_value_counts = dict(sorted(topic_counts.items(), key=lambda x: x[1], reverse=True))
    
    return sorted_value_counts

import re

def find_matching_strings(df, column_name, search_string):

    pattern = r'\b(' + '|'.join([re.escape(term) for term in search_string]) + r')\w*\b'
    
    # Inicializar um conjunto para armazenar os resultados únicos
    unique_strings = set()

    # Iterar sobre as listas de tópicos no DataFrame
    for topics in df[column_name]:
        for topic in topics:
            # Se algum dos termos for encontrado, adicionar à lista
            if re.search(pattern, topic, re.IGNORECASE):
                unique_strings.add(topic)
    
    return unique_strings


def replace_with_string(df, original_column, new_column, search_strings_set, string_name):
    # Cria uma nova coluna com a mesma estrutura da original
    df[new_column] = df[original_column].apply(
        lambda topics: [string_name if topic in search_strings_set else topic for topic in topics]
    )
    return df

def plot_string_occurrences_by_year(df, column_name, search_string):
    
    df[column_name] = df[column_name].apply(lambda topics: sum(search_string in topic for topic in topics))

    year_counts = df.groupby('year')[column_name].sum()

    bar_color = generate_random_color()

    plt.figure(figsize=(12, 7))
    plt.bar(year_counts.index, year_counts.values, color=bar_color)

    plt.title(f'Ocurrencias de <{search_string}> por Año', fontsize=16)
    plt.xlabel('Año', fontsize=14, labelpad=20)
    plt.ylabel(f'Número de Ocurrencias de <{search_string}>', fontsize=14, labelpad=20)

    for year, count in year_counts.items():
        plt.text(year, count, str(count), ha='center', va='bottom', fontsize=12)

    plt.xticks(year_counts.index, rotation=45, ha='right', fontsize=12)
    
    plt.tight_layout()

    plt.show()


def plot_replaced_string(df, column_name, search_strings, new_name):
    new_column_name = f"{new_name}_topics"
    
    search_strings = [search_strings] if isinstance(search_strings, str) else search_strings

    topics_set = find_matching_strings(df, column_name, search_strings)
    print(f"Valores encontrados en <{column_name}> correspondientes a {search_strings}: {topics_set}")

    df = replace_with_string(df, column_name, new_column_name, topics_set, new_name)

    replaced_set = find_matching_strings(df, new_column_name, [new_name])
    print(f"Valores encontrados en <{new_column_name}> correspondientes a {new_name}: {replaced_set}")

    print(f"\nNúmero de valores correspondientes a <{search_strings}> en <{column_name}>: {len(topics_set)}")
    print(f"Número de valores correspondientes a <{new_name}> en <{new_column_name}>: {len(replaced_set)}")

    plot_string_occurrences_by_year(df, new_column_name, new_name)


def plot_occurrences_by_year(df, column_name, p_language):

    df_filtered = df[df[column_name].str.lower() == p_language.lower()]
    
    if df_filtered.empty:
        print(f'Nenhuma ocorrência de "{p_language}" encontrada.')
        return
    
    year_counts = df_filtered.groupby('year').size()
    bar_color = generate_random_color()

    plt.figure(figsize=(10, 6))
    ax = year_counts.plot(kind='bar', color=bar_color)
    
    for i, count in enumerate(year_counts):
        ax.text(i, count + 0.1, str(count), ha='center', va='bottom', fontsize=12)
    
    plt.title(f'Ocurrencias de <{p_language}> por Año', fontsize=16)
    plt.xlabel('Año', fontsize=14, labelpad=20)
    plt.ylabel(f'Número de Ocurrencias de <{p_language}>', fontsize=14, labelpad=20)
    
    plt.xticks(rotation=45, ha='right', fontsize=12)
    
    plt.tight_layout()
    
    plt.show()

