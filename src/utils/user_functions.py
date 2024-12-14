from datetime import datetime, date
import json
import requests
import time

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