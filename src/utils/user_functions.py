# Importación de bibliotecas estándar

from collections import Counter
from datetime import datetime, date
from itertools import combinations
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, pearsonr
import os
import pandas as pd
import random
import requests
import seaborn as sns
import time


# 1. FUNCIONES PARA ACCEDER A LA API REST DE GITHUB Y OBTENER LOS DATOS EN getting_GH_datasets.ipynb

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


# -------------------------------------------------------------------------------------------------------------------------------------


# 2. FUNCIONES PARA AUTOMATIZAR CÁLCULOS Y PLOTS DE GRÁFICOS DEL NOTEBOOK main.ipynb


def filter_repository_data(items, columns):
    filtered_items = []
    for item in items:
        filtered_item = {col: item.get(col) for col in columns}
        filtered_items.append(filtered_item)
    return filtered_items

def count_data_by_year(df, column_datetime):
    year_counts = df['year'].value_counts().sort_index()
    return year_counts.to_dict()

def plot_data_by_year(df, column_datetime, color='cornflowerblue'):
    
    year_counts = df['year'].value_counts().sort_index()

    plt.figure(figsize=(12, 7))
    bars = plt.bar(year_counts.index, year_counts.values, color=color)

    plt.title('Número de Repositórios por Año (2010-2024)', fontsize=16, pad=20)
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

def generate_random_color_list(num_colors):
    colors = []
    for _ in range(num_colors):
        r = random.randint(10, 140)
        g = random.randint(10, 140)
        b = random.randint(10, 180)
        color = "#{:02x}{:02x}{:02x}".format(r, g, b)
        colors.append(color)
    
    return colors

def categorical_distribution_aspect(df, columnas_categoricas, relativa=False, mostrar_valores=False):
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
    
    # Inicializar un conjunto para almacenar los resultados únicos
    unique_strings = set()

    # Iterar sobre las listas de temas en el DataFrame
    for topics in df[column_name]:
        for topic in topics:
            # Si se encuentra alguno de los términos, añadirlo a la lista
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


def plot_occurrences_by_year(df, column_name, search_string, year_range=None):
    
    # Filtra el DataFrame para el lenguaje buscado
    df_filtered = df[df[column_name].str.lower() == search_string.lower()]
    
    if df_filtered.empty:
        print(f'No se encontraron ocurrencias de "{search_string}".')
        return
    
    # Si no se proporciona un rango de años, se usa todos los años disponibles
    if year_range is None:
        year_range = range(df['year'].min(), df['year'].max() + 1)
    
    # Filtra los datos para el intervalo de años proporcionado
    df_filtered = df_filtered[df_filtered['year'].isin(year_range)]
    
    # Contar las ocurrencias por año para el lenguaje específico
    year_counts_language = df_filtered.groupby('year').size()

    # Contar los datos totales por año en el DataFrame
    total_counts_by_year = count_data_by_year(df, 'year')
    
    # Calcular los porcentajes para cada año
    percentage_counts = {year: (count / total_counts_by_year.get(year, 1)) * 100
                         for year, count in year_counts_language.items()}
    
    # Generar una lista de colores aleatorios para cada barra
    num_colors = len(year_counts_language)
    bar_colors = generate_random_color_list(num_colors)
    
    # Crear los gráficos
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    # Gráfico 1: Ocurrencias Absolutas
    axes[0].bar(year_counts_language.index, year_counts_language.values, color=bar_colors)
    axes[0].set_title(f'Ocurrencias de "{search_string}" por Año (Valores Absolutos)', fontsize=14)
    axes[0].set_xlabel('Año', fontsize=12)
    axes[0].set_ylabel('Número de Ocurrencias', fontsize=12)
    axes[0].set_xticks(year_counts_language.index)
    axes[0].tick_params(axis='x', rotation=45)

    # Añadir los valores en la parte superior de las barras en el gráfico 1
    for i, count in enumerate(year_counts_language):
        axes[0].text(year_counts_language.index[i], count + 0.1, str(count), ha='center', va='bottom', fontsize=10)

    # Gráfico 2: Porcentaje Normalizado por Año
    axes[1].bar(percentage_counts.keys(), percentage_counts.values(), color=bar_colors)
    axes[1].set_title(f'Ocurrencias de "{search_string}" por Año (Valores Porcentuales)', fontsize=14)
    axes[1].set_xlabel('Año', fontsize=12)
    axes[1].set_ylabel('Porcentaje de Ocurrencias (%)', fontsize=12)
    axes[1].set_xticks(list(percentage_counts.keys()))
    axes[1].tick_params(axis='x', rotation=45)

    # Añadir los valores porcentuales en la parte superior de las barras en el gráfico 2
    for i, count in enumerate(percentage_counts.values()):
        axes[1].text(list(percentage_counts.keys())[i], count, f'{count:.2f}%', ha='center', va='bottom', fontsize=10)
    
    # Mostrar los gráficos
    plt.show()


def plot_string_with_normalization(df, column_name, search_strings, new_name, year_range=None):
    # Crear una nueva columna sustituyendo las cadenas encontradas
    new_column_name = f"{new_name}_topics"

    search_strings = [search_strings] if isinstance(search_strings, str) else search_strings

    topics_set = find_matching_strings(df, column_name, search_strings)
    print(f"Valores encontrados en <{column_name}> correspondientes a {search_strings}: {topics_set}")

    df = replace_with_string(df, column_name, new_column_name, topics_set, new_name)

    replaced_set = find_matching_strings(df, new_column_name, [new_name])
    print(f"Valores encontrados en <{new_column_name}> correspondientes a {new_name}: {replaced_set}")

    print(f"\nNúmero de valores correspondientes a <{search_strings}> en <{column_name}>: {len(topics_set)}")
    print(f"Número de valores correspondientes a <{new_name}> en <{new_column_name}>: {len(replaced_set)}")

    # Filtrar los datos por intervalo de años, si se proporciona
    if year_range:
        df = df[df['year'].isin(year_range)]

    # Contar solo una ocurrencia única de 'new_name' por celda
    df.loc[:, new_column_name] = df[new_column_name].apply(lambda topics: 1 if new_name in set(topics) else 0)
    year_counts = df.groupby('year')[new_column_name].sum()

    # Contar el total por año para la normalización
    total_counts_by_year = count_data_by_year(df, 'year')

    # Valores normalizados porcentualmente
    normalized_counts = {year: (year_counts.get(year, 0) / total_counts_by_year[year]) * 100 for year in total_counts_by_year}

    # Crear una figura con dos subgráficos
    fig, axes = plt.subplots(1, 2, figsize=(20, 6), constrained_layout=True)

    # Subgráfico 1: Gráfico de valores absolutos
    bar_colors = generate_random_color_list(len(year_counts))
    axes[0].bar(year_counts.index, year_counts.values, color=bar_colors)
    axes[0].set_title(f'Ocurrencias de <{new_name}> por Año', fontsize=16)
    axes[0].set_xlabel('Año', fontsize=14, labelpad=10)
    axes[0].set_ylabel(f'Número de Ocurrencias de <{new_name}>', fontsize=14, labelpad=10)
    axes[0].grid(axis='y', linestyle='--', alpha=0.6)

    # Añadir etiquetas de texto en el gráfico de valores absolutos
    for i, (year, count) in enumerate(year_counts.items()):
        axes[0].text(year, count, str(count), ha='center', va='bottom', fontsize=12, color=bar_colors[i])

    # Subgráfico 2: Gráfico de valores normalizados
    bar_colors_2 = generate_random_color_list(len(normalized_counts))
    axes[1].bar(normalized_counts.keys(), normalized_counts.values(), color=bar_colors_2)
    axes[1].set_title(f'Ocurrencias Normalizadas de <{new_name}> por Año (%)', fontsize=16)
    axes[1].set_xlabel('Año', fontsize=14, labelpad=10)
    axes[1].set_ylabel(f'Porcentaje de Ocurrencias de <{new_name}>', fontsize=14, labelpad=10)
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)

    # Añadir etiquetas de texto en el gráfico de valores normalizados
    for i, (year, percentage) in enumerate(normalized_counts.items()):
        axes[1].text(year, percentage, f"{percentage:.2f}%", ha='center', va='bottom', fontsize=12, color=bar_colors_2[i])

    # Ajustar las etiquetas del eje x
    for ax in axes:
        ax.set_xticks(list(year_counts.index))
        ax.set_xticklabels(year_counts.index, rotation=45, ha='right', fontsize=12)

    plt.show()


def plot_string_with_normalization_including_all(df, column_name, search_strings, new_name, year_range=None):    # Conta valores repetidos!
    # Crear una nueva columna reemplazando las cadenas encontradas 
    new_column_name = f"{new_name}_topics"

    search_strings = [search_strings] if isinstance(search_strings, str) else search_strings

    topics_set = find_matching_strings(df, column_name, search_strings)
    print(f"Valores encontrados en <{column_name}> correspondientes a {search_strings}: {topics_set}")

    df = replace_with_string(df, column_name, new_column_name, topics_set, new_name)

    replaced_set = find_matching_strings(df, new_column_name, [new_name])
    print(f"Valores encontrados en <{new_column_name}> correspondientes a {new_name}: {replaced_set}")

    print(f"\nNúmero de valores correspondientes a <{search_strings}> en <{column_name}>: {len(topics_set)}")
    print(f"Número de valores correspondientes a <{new_name}> en <{new_column_name}>: {len(replaced_set)}")

    # Filtrar los datos por el intervalo de años, si se proporciona
    if year_range:
        df = df[df['year'].isin(year_range)]

    # Contar las ocurrencias por año
    df.loc[:, new_column_name] = df[new_column_name].apply(lambda topics: sum(new_name in topic for topic in topics))
    year_counts = df.groupby('year')[new_column_name].sum()

    # Contar el total por año para la normalización
    total_counts_by_year = count_data_by_year(df, 'year')

    # Valores normalizados porcentualmente
    normalized_counts = {year: (year_counts.get(year, 0) / total_counts_by_year[year]) * 100 for year in total_counts_by_year}

    # Crear una figura con dos subgráficos
    fig, axes = plt.subplots(1, 2, figsize=(20, 6), constrained_layout=True)

    # Generar una lista de colores para los años
    unique_years = list(year_counts.index)
    bar_colors = generate_random_color_list(len(unique_years))

    # Subgráfico 1: Gráfico de valores absolutos
    axes[0].bar(unique_years, year_counts.values, color=bar_colors)
    axes[0].set_title(f'Ocurrencias de <{new_name}> por Año', fontsize=16)
    axes[0].set_xlabel('Año', fontsize=14, labelpad=10)
    axes[0].set_ylabel(f'Número de Ocurrencias de <{new_name}>', fontsize=14, labelpad=10)
    axes[0].grid(axis='y', linestyle='--', alpha=0.6)

    # Añadir etiquetas de texto al gráfico de valores absolutos
    for i, (year, count) in enumerate(year_counts.items()):
        axes[0].text(year, count, str(count), ha='center', va='bottom', fontsize=12, color=bar_colors[i])

    # Subgráfico 2: Gráfico de valores normalizados
    axes[1].bar(unique_years, [normalized_counts[year] for year in unique_years], color=bar_colors)
    axes[1].set_title(f'Ocurrencias Normalizadas de <{new_name}> por Año (%)', fontsize=16)
    axes[1].set_xlabel('Año', fontsize=14, labelpad=10)
    axes[1].set_ylabel(f'Porcentaje de Ocurrencias de <{new_name}>', fontsize=14, labelpad=10)
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)

    # Añadir etiquetas de texto al gráfico de valores normalizados
    for i, (year, percentage) in enumerate(normalized_counts.items()):
        axes[1].text(year, percentage, f"{percentage:.2f}%", ha='center', va='bottom', fontsize=12, color=bar_colors[i])

    # Ajustar las etiquetas del eje x
    for ax in axes:
        ax.set_xticks(unique_years)
        ax.set_xticklabels(unique_years, rotation=45, ha='right', fontsize=12)

    plt.show()


def analyze_topic_success(df, topics_column, search_string, stargazers_columns, log_normalization=False, oversample=False):
    """
    Analiza la relación entre la presencia de un tema específico y el éxito del repositorio,
    realizando una prueba de Mann-Whitney con opción de sobreamostrado y normalización estándar o logarítmica.

    :param df: DataFrame que contiene los datos de los repositorios.
    :param topics_column: Nombre de la columna que contiene la lista de temas.
    :param search_string: Tema específico a analizar.
    :param stargazers_columns: Lista de columnas numéricas de éxito.
    :param log_normalization: Si es True, aplica normalización logarítmica.
    :param oversample: Si es True, aplica sobreamostrado para igualar el tamaño de los grupos.
    """
    # Filtra los datos con y sin la etiqueta
    df_with_tag = df[df[topics_column].apply(lambda x: search_string in x)]
    df_without_tag = df[~df[topics_column].apply(lambda x: search_string in x)]

    print(f"Número de repositorios con la etiqueta '{search_string}': {len(df_with_tag)}")
    print(f"Número de repositorios sin la etiqueta '{search_string}': {len(df_without_tag)}")

    # Aplica sobreamostrado si oversample=True
    if oversample:
        df_with_tag = df_with_tag.sample(len(df_without_tag), replace=True, random_state=42)
        print(f"Número de repositorios con la etiqueta '{search_string}' (después de sobreamostrado): {len(df_with_tag)}")

    fig, axes = plt.subplots(1, len(stargazers_columns), figsize=(16, 6))

    p_values = []
    for i, stargazers_column in enumerate(stargazers_columns):
        # Normalización
        if log_normalization:
            sum_with_tag = np.log1p(df_with_tag[stargazers_column]).sum()
            sum_without_tag = np.log1p(df_without_tag[stargazers_column]).sum()
        else:
            sum_with_tag = df_with_tag[stargazers_column].sum()
            sum_without_tag = df_without_tag[stargazers_column].sum()
        
        count_with_tag = len(df_with_tag)
        count_without_tag = len(df_without_tag)

        normalized_with_tag = sum_with_tag / count_with_tag if count_with_tag > 0 else 0
        normalized_without_tag = sum_without_tag / count_without_tag if count_without_tag > 0 else 0

        categories = [f'Con {search_string}', f'Sin {search_string}']
        values = [normalized_with_tag, normalized_without_tag]

        axes[i].bar(categories, values, color=['#800080', '#FFB300'])
        axes[i].set_title(f'Comparación de {stargazers_column} Normalizados\ncon y sin la etiqueta "{search_string}"', fontsize=14)
        axes[i].set_xlabel('Grupo', fontsize=12)
        axes[i].set_ylabel(f'{stargazers_column} Normalizados', fontsize=12)

        # Prueba de Mann-Whitney
        _, p_value = mannwhitneyu(df_with_tag[stargazers_column], df_without_tag[stargazers_column])
        p_values.append(p_value)

        axes[i].text(0.25, 0.05, f'Valor p = {p_value:.4f}', ha='center', va='center', fontsize=15, color='#FFFFFF', transform=axes[i].transAxes)

        # Prints adicionales con los valores totales y normalizados
        print(f"\nPara {stargazers_column}:")
        print(f"Total 'con {search_string}': {sum_with_tag}")
        print(f"Total 'sin {search_string}': {sum_without_tag}")
        print(f"Valor normalizado 'con {search_string}': {normalized_with_tag}")
        print(f"Valor normalizado 'sin {search_string}': {normalized_without_tag}")

    plt.tight_layout()
    plt.show()

    for i, stargazers_column in enumerate(stargazers_columns):
        print(f"Valor p de la prueba de Mann-Whitney para {stargazers_column}: {p_values[i]:.4f}")
        alpha = 0.05
        if p_values[i] < alpha:
            print(f"Rechazamos la hipótesis nula para {stargazers_column}: Existe una diferencia significativa entre los dos grupos.\n")
        else:
            print(f"No rechazamos la hipótesis nula para {stargazers_column}: No existe una diferencia significativa.\n")


def normalize_by_months(df):
    """
    Normaliza los valores de stargazers_count y forks_count inversamente al número de meses desde la creación,
    para reducir el impacto del tiempo en repositorios más antiguos y aumentar la relevancia de repositorios más nuevos.
    
    Parámetros:
        df (pd.DataFrame): DataFrame que contiene los datos.
    
    Retorna:
        pd.DataFrame: DataFrame con las columnas normalizadas añadidas.
    """
    # Convierte las columnas de fecha a datetime
    df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_localize(None)  # Elimina la zona horaria
    
    # Determina el intervalo de fechas usando la columna 'year'
    min_year = df['year'].min()
    max_year = df['year'].max()
    
    # Define el primer día del año mínimo y el último día del año máximo
    start_date = pd.to_datetime(f'{min_year}-01-01')
    end_date = pd.to_datetime(f'{max_year}-12-31')
    
    # Filtra los repositorios en el período deseado
    df_filtered = df[(df['created_at'] >= start_date) & (df['created_at'] <= end_date)].copy()
    
    # Calcula el número de meses desde la creación hasta el final del período
    df_filtered['months_since_creation'] = (
        (end_date.year - df_filtered['created_at'].dt.year) * 12 +
        (end_date.month - df_filtered['created_at'].dt.month)
    )
    
    # Normaliza inversamente al tiempo (meses desde la creación)
    # Usamos la raíz cuadrada para reducir el impacto del tiempo de forma suave (puede ajustarse)
    df_filtered['stargazers_rate'] = df_filtered['stargazers_count'] / np.sqrt(df_filtered['months_since_creation'] + 1)
    df_filtered['forks_rate'] = df_filtered['forks_count'] / np.sqrt(df_filtered['months_since_creation'] + 1)
    
    return df_filtered



def normalize_by_days(df):
    """
    Normaliza los valores de stargazers_count y forks_count por el número de días desde la creación,
    considerando los repositorios creados entre el primer y el último año presentes en la columna 'year'.
    
    Parámetros:
        df (pd.DataFrame): DataFrame que contiene los datos.
    
    Retorna:
        pd.DataFrame: DataFrame con las columnas normalizadas añadidas.
    """
    # Convierte las columnas de fecha a datetime
    df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_localize(None)  # Elimina la zona horaria
    
    # Determina el intervalo de fechas usando la columna 'year'
    min_year = df['year'].min()
    max_year = df['year'].max()
    
    # Define el primer día del año mínimo y el último día del año máximo
    start_date = pd.to_datetime(f'{min_year}-01-01')
    end_date = pd.to_datetime(f'{max_year}-12-31')
    
    # Filtra los repositorios en el período deseado
    df_filtered = df[(df['created_at'] >= start_date) & (df['created_at'] <= end_date)].copy()
    
    # Calcula el número de días desde la creación hasta el final del período
    df_filtered['days_since_creation'] = (end_date - df_filtered['created_at']).dt.days
    
    # Normaliza los valores
    df_filtered['stargazers_rate'] = df_filtered['stargazers_count'] / (df_filtered['days_since_creation'] + 1)
    df_filtered['forks_rate'] = df_filtered['forks_count'] / (df_filtered['days_since_creation'] + 1)
    
    return df_filtered

def detect_outliers(df, numeric_column, factor=1.5):
    """
    Detecta los outliers en una columna numérica utilizando el método IQR (Rango Intercuartil).
    Devuelve solo el límite superior para considerar los outliers.

    :param df: DataFrame que contiene los datos.
    :param numeric_column: Nombre de la columna numérica a analizar (ej: 'stargazers_count').
    :param factor: Factor multiplicador para el cálculo del intervalo de outliers (por defecto: 1.5).
    :return: Límite superior a partir del cual un valor es considerado un outlier.
    """
    # Calculando Q1 (primer cuartil) y Q3 (tercer cuartil)
    Q1 = df[numeric_column].quantile(0.25)
    Q3 = df[numeric_column].quantile(0.75)
    
    # Calculando el intervalo intercuartil (IQR)
    IQR = Q3 - Q1
    
    # Definiendo el límite superior para identificar outliers
    upper_limit = Q3 + factor * IQR
    
    # Devuelve el límite superior
    return upper_limit

def plot_scatter_with_regression(df, columns):
    """
    Genera gráficos de dispersión para todas las combinaciones posibles de columnas numéricas,
    añade una línea de regresión y muestra la correlación de Pearson.

    Parámetros:
    - df: DataFrame con los datos.
    - columns: Lista de columnas numéricas a analizar.
    """
    # Configurar estilo de los gráficos
    sns.set(style='whitegrid')

    # Crear todas las combinaciones posibles de 2 columnas
    col_pairs = list(combinations(columns, 2))
    num_plots = len(col_pairs)

    # Definir el tamaño de la figura
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 18))  # 3 filas y 2 columnas

    # Ajustar los espacios entre los subgráficos
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    # Para cada par de columnas, crear un gráfico de dispersión con línea de regresión
    for i, (col_x, col_y) in enumerate(col_pairs):
        row = i // 2  # Determinar fila
        col = i % 2   # Determinar columna
        
        ax = axes[row, col]
        sns.regplot(x=df[col_x], y=df[col_y], scatter_kws={"s": 5}, line_kws={"color": "red", "lw": 1}, ax=ax)
        
        # Calcular la correlación de Pearson
        corr, _ = pearsonr(df[col_x], df[col_y])
        
        # Imprimir la correlación
        print(f"Correlación de Pearson entre {col_x} y {col_y}: {corr:.2f}")
        
        # Título con la correlación
        ax.set_title(f'{col_x} vs {col_y}\nPearson: {corr:.2f}', fontsize=14)
        ax.set_xlabel(col_x, fontsize=12)
        ax.set_ylabel(col_y, fontsize=12)

    plt.show()

def top_20_strings_by_columns(df, col1, col2, column_topic):
    """
    Genera gráficos de barras para los 20 temas más comunes en una columna categórica, 
    considerando la suma de valores de dos columnas numéricas de forma independiente.

    Parámetros:
    - df: DataFrame con los datos.
    - col1, col2: Nombres de las columnas numéricas para calcular la suma.
    - column_topic: Nombre de la columna categórica que contiene listas de cadenas.
    """

    def get_top_20_topics_by_sum(df, numeric_col, topics_col):
        # Diccionario para almacenar la suma de los valores para cada tema
        topic_sums = Counter()

        # Itera sobre cada fila del DataFrame
        for _, row in df.iterrows():
            if isinstance(row[topics_col], list):  # Verifica si el valor es una lista
                for topic in row[topics_col]:
                    topic_sums[topic] += row[numeric_col]  # Suma el valor de la columna numérica

        # Ordena los temas por la suma en orden descendente
        top_20_topics = topic_sums.most_common(20)
        if not top_20_topics:  # En caso de que no existan temas
            return [], []
        return zip(*top_20_topics)  # Separa los temas y los recuentos

    # Obtiene los datos para ambas columnas
    topics1, counts1 = get_top_20_topics_by_sum(df, col1, column_topic)
    topics2, counts2 = get_top_20_topics_by_sum(df, col2, column_topic)

    # Crea la figura y los ejes
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    if topics1 and counts1:
        axes[0].barh(topics1, counts1, color='royalblue')
        axes[0].set_title(f'Top 20 Temas en <{col1}>', fontsize=16)
        axes[0].set_xlabel('Suma', fontsize=14)
        axes[0].invert_yaxis()  # Invierte el eje Y en el gráfico de la izquierda

    if topics2 and counts2:
        axes[1].barh(topics2, counts2, color='darkgreen')
        axes[1].set_title(f'Top 20 Tópicos en <{col2}>', fontsize=16)
        axes[1].set_xlabel('Suma', fontsize=14)

    # Ajusta el diseño
    plt.tight_layout()
    plt.show()


def boxplot_generate(df, columna):

    if columna not in df.columns:
        raise ValueError(f"La columna '{columna}' no está presente en el DataFrame.")
    if not (df[columna].dtype.kind in 'biufc'):
        raise ValueError(f"La columna '{columna}' no es numérica.")
    
    plt.figure(figsize=(18, 6))
    sns.boxplot(x=df[columna], color='skyblue')
    plt.title(f'Boxplot de {columna}', fontsize=14)
    plt.xlabel(columna, fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()

