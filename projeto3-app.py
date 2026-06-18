# Projeto 3 - Implementação de Feature Store em Pipeline de Deploy de Machine Learning
# Módulo de Deploy do Modelo via API

# Imports
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Cria a app
app = Flask(__name__)

# Caminhos para o modelo treinado e o scaler
MODEL_PATH = 'dsa_ml_pipeline/modelo_dsa.pkl'
SCALER_PATH = 'dsa_ml_pipeline/scaler.pkl'  

# Bloco try/except para carregar o modelo treinado
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None
    print("Erro: Modelo não encontrado.")

# Bloco try/except para carregar o scaler salvo
try:
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError:
    scaler = None
    print("Erro: Scaler não encontrado.")

# Endpoint para prever novos dados
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obter os dados do request
        data = request.get_json()

        if not data:
            return jsonify({'error': 'Nenhum dado fornecido'}), 400

        # Converter para DataFrame do pandas
        df = pd.DataFrame(data)

        # Remover colunas desnecessárias
        if 'entity_id' in df.columns:
            df = df.drop(columns=['entity_id'])

        if 'target' in df.columns:
            df = df.drop(columns=['target'])

        # Verificar se o scaler foi carregado
        if scaler:
            # Aplicar o scaler às features numéricas
            df[['feature1', 'feature2']] = scaler.transform(df[['feature1', 'feature2']])
        else:
            return jsonify({'error': 'Scaler não carregado'}), 500

        # Verificar se o modelo foi carregado
        if model:
            # Fazer a predição
            predictions = model.predict(df)
            return jsonify({'predictions': predictions.tolist()})
        else:
            return jsonify({'error': 'Modelo não carregado'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)
