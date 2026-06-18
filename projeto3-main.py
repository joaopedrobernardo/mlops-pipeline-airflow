# Projeto 3 - Implementação de Feature Store em Pipeline de Deploy de Machine Learning
# Módulo Principal

# Imports
import os
from dsa_feature_store.feature_engineering import dsa_carrega_dados, dsa_cria_atributos, dsa_salva_atributos
from dsa_feature_store.feature_store import FeatureStore
from dsa_ml_pipeline.model_training import dsa_treina_modelo
from dsa_ml_pipeline.model_inference import dsa_inferencia

# Caminhos de arquivos
RAW_DATA_PATH = 'dsa_dados/dados_brutos.csv'
FEATURE_STORE_PATH = 'dsa_dados/feature_store.csv'
MODEL_PATH = 'dsa_ml_pipeline/modelo_dsa.pkl'

# Bloco principal
def main():

    # Ingestão dos dados brutos
    raw_data = dsa_carrega_dados(RAW_DATA_PATH)

    # Criação dos atributos
    features = dsa_cria_atributos(raw_data)
    
    # Cria instância da Feature Store
    feature_store = FeatureStore(FEATURE_STORE_PATH)

    # Salva os dados processados na Feature Store
    feature_store.dsa_salva_atributos(features)

    # Treina o modelo com as features armazenadas na Feature Store
    # Observe que a variável target não fica na Feature Store
    labels = raw_data['target']  
    dsa_treina_modelo(labels, MODEL_PATH)

    # Carrega os atributos da Feature Store para inferência
    new_features = feature_store.dsa_carrega_atributos()

    # Faz a inferência extraindo as previsões com os atributos da feature store
    predictions = dsa_inferencia(MODEL_PATH, new_features)
    print(f"Previsões: {predictions}")

if __name__ == '__main__':
    main()
