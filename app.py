from flask import Flask, request, jsonify, render_template
import pickle as pkl
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__, static_url_path='/Source', static_folder='Source')

# Cargar el modelo entrenado y los diccionarios de vocabulario
with open('model.pkl', 'rb') as f:
    model = pkl.load(f)

with open('team_vocab.pkl', 'rb') as f:
    team_classes = pkl.load(f)

with open('venue_vocab.pkl', 'rb') as f:
    venue_classes = pkl.load(f)

with open('opponent_vocab.pkl', 'rb') as f:
    opponent_classes = pkl.load(f)

# Inicializar los encoders con los vocabularios cargados
team_encoder = LabelEncoder()
team_encoder.classes_ = team_classes

venue_encoder = LabelEncoder()
venue_encoder.classes_ = venue_classes

opponent_encoder = LabelEncoder()
opponent_encoder.classes_ = opponent_classes

def predict_winner(params):
    # Convertir los inputs a valores numéricos
    team_encoded = team_encoder.transform([params['team']])[0]
    venue_encoded = venue_encoder.transform([params['venue']])[0]
    opponent_encoded = opponent_encoder.transform([params['opponent']])[0]
    
    # Crear el array de features
    features = np.array([
        team_encoded,
        opponent_encoded,
        venue_encoded,
        params['gf'],
        params['ga'],
        params['poss'],
        params['sh'],
        params['sot'],
        params['dist']
    ]).reshape(1, -1)
    
    # Hacer la predicción
    prediction = model.predict(features)
    
    # Convertir la predicción numérica a la etiqueta correspondiente
    result = "Win" if prediction[0] == 2 else "Lose"
    
    return result

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    params = {
        'team': request.form['team'],
        'venue': request.form['venue'],
        'opponent': request.form['opponent'],
        'gf': int(request.form['gf']),
        'ga': int(request.form['ga']),
        'poss': int(request.form['poss']),
        'sh': int(request.form['sh']),
        'sot': int(request.form['sot']),
        'dist': int(request.form['dist'])
    }
    
    prediction = predict_winner(params)
    
    return render_template('index.html', prediction_text=f'Resultado predicho: {prediction}')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
