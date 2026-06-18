# Projeto 3 - Implementação de Feature Store em Pipeline de Deploy de Machine Learning
# Módulo de Treinamento do Modelo

# Imports
import joblib
import pandas as pd
from dsa_feature_store.feature_store import FeatureStore
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Caminho da feature store
FEATURE_STORE_PATH = 'dsa_dados/feature_store.csv'

# Função para carregar dados da feature store e separá-los em features e labels
def dsa_carrega_dados_feature_store(feature_store_path, target):

    # Extrai os dados da feature store
    feature_store = FeatureStore(feature_store_path)

    # Carrega os atributos
    features = feature_store.dsa_carrega_atributos()

    # A variável alvo não fica na feature store
    labels = target

    # Retorna os dados
    return features, labels

# Função de treinamento do modelo
def dsa_treina_modelo(target, model_output_path):

    # Carrega os dados da feature store
    features, labels = dsa_carrega_dados_feature_store(FEATURE_STORE_PATH, target)
    
    # Divisão dos dados
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.3, random_state = 42)

    # Definição e treinamento do modelo
    model = RandomForestClassifier(n_estimators = 100, random_state = 42)
    model.fit(X_train, y_train)

    # Avaliação
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Acurácia do Modelo com Dados de Teste: {accuracy}')

    # Salvar o modelo treinado
    joblib.dump(model, model_output_path)

