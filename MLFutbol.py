import pandas as pd
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Cargar los datos
matches_path = 'matches_normalized.csv'
matches_df = pd.read_csv(matches_path)

# Seleccionar las columnas relevantes
features = ['team', 'opponent', 'venue', 'gf', 'ga', 'poss', 'sh', 'sot', 'dist']
target = 'result'

# Verificar la distribución de clases
print("Distribución de clases en el dataset:")
print(matches_df[target].value_counts())

# Codificar las etiquetas del objetivo
label_encoder = LabelEncoder()
matches_df[target] = label_encoder.fit_transform(matches_df[target])

# Codificar las características categóricas
team_encoder = LabelEncoder()
opponent_encoder = LabelEncoder()
venue_encoder = LabelEncoder()

matches_df['team'] = team_encoder.fit_transform(matches_df['team'])
matches_df['opponent'] = opponent_encoder.fit_transform(matches_df['opponent'])
matches_df['venue'] = venue_encoder.fit_transform(matches_df['venue'])

# Dividir los datos en características y objetivo
X = matches_df[features]
y = matches_df[target]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy:.2f}')

# Mostrar el reporte de clasificación
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))

# Guardar el modelo y los encoders
with open('model.pkl', 'wb') as f:
    pkl.dump(model, f)

with open('team_vocab.pkl', 'wb') as f:
    pkl.dump(team_encoder.classes_, f)

with open('opponent_vocab.pkl', 'wb') as f:
    pkl.dump(opponent_encoder.classes_, f)

with open('venue_vocab.pkl', 'wb') as f:
    pkl.dump(venue_encoder.classes_, f)

print("Modelo entrenado y guardado exitosamente.")
