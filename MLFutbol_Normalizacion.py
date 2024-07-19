import pandas as pd

# Cargar los datos
matches_path = 'matches.csv'
data = pd.read_csv(matches_path)

# Diccionario de normalización basado en los datos proporcionados
normalization_dict = {
    'Manchester Utd': 'Manchester United',
    'Newcastle Utd': 'Newcastle United',
    'Brighton and Hove Albion': 'Brighton',
    'West Bromwich Albion': 'West Brom',
    'Tottenham Hotspur': 'Tottenham',
    'Wolverhampton Wanderers': 'Wolves',
    'Sheffield Utd': 'Sheffield United',
    'Man City': 'Manchester City',  # Agregando más normalizaciones si es necesario
    'Spurs': 'Tottenham',
    'Man United': 'Manchester United'
}

# Función para normalizar los nombres de los equipos
def normalize_team_name(name):
    return normalization_dict.get(name, name)

# Aplicar la normalización a las columnas 'team' y 'opponent'
data['team'] = data['team'].apply(normalize_team_name)
data['opponent'] = data['opponent'].apply(normalize_team_name)

# Guardar el archivo normalizado en un nuevo CSV
output_file_path = 'matches_normalized.csv'
data.to_csv(output_file_path, index=False)

print(f'Archivo guardado en: {output_file_path}')
