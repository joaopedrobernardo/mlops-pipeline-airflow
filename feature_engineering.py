# Projeto 3 - Implementação de Feature Store em Pipeline de Deploy de Machine Learning
# Módulo de Engenharia de Atributos

# Imports
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib  # Import necessário para salvar o scaler

# Ingestão de dados
def dsa_carrega_dados(file_path):
    data = pd.read_csv(file_path)
    return data

# Função de engenharia de atributos
def dsa_cria_atributos(df, scaler_path = 'dsa_ml_pipeline/scaler.pkl'):

    # Criando uma feature de "diferença de tempo"
    df['time_diff'] = pd.to_datetime(df['end_time']) - pd.to_datetime(df['start_time'])
    df['time_diff'] = df['time_diff'].dt.total_seconds()

    # Padronização das features quantitativas
    scaler = StandardScaler()
    df[['feature1', 'feature2']] = scaler.fit_transform(df[['feature1', 'feature2']])

    # Salvar o scaler em disco
    joblib.dump(scaler, scaler_path)
    print(f"Scaler salvo em: {scaler_path}")

    # Remove colunas desnecessárias
    df = df.drop(columns=['start_time', 'end_time', 'entity_id', 'target'], errors='ignore')

    return df

# Função para salvar as features processadas
def dsa_salva_atributos(df, file_path):
    df.to_csv(file_path, index=False)
