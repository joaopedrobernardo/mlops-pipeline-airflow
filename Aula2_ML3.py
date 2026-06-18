# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 14:36:33 2025

@author: mathe
"""

# -*- coding: utf-8 -*-
# MLP para o dataset "diabetes" (regressão) com Optuna + MLflow (uso local)
# Rode este arquivo no Spyder. Depois, no terminal:  mlflow ui  (e abra http://127.0.0.1:5000)

import os
import json
import numpy as np
import tensorflow as tf
import optuna
import mlflow
import mlflow.keras

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------
# Configurações gerais
# ---------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Diretório local para os runs (pasta "mlruns" ao lado do script)
here = os.path.abspath(os.path.dirname(__file__)) if "__file__" in globals() else os.getcwd()  #pega a pasta onde o arquivo está salvo
mlruns_dir = os.path.join(here, "mlruns")  #cria uma pasta na mesma onde está o arquivo pyhon
mlflow.set_tracking_uri(f"file:///{mlruns_dir.replace(os.sep, '/')}")  # Windows-safe...  é o endereço onde salva as runs
mlflow.set_experiment("mlp_diabetes_optuna_local")  # define o nome do experimento

# ---------------------------
# Dados
# ---------------------------
X, y = load_diabetes(return_X_y=True)  # problema de regressão (alvo contínuo)
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.25, random_state=SEED)
X_val, X_test, y_val, y_test   = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=SEED)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

input_dim = X_train.shape[1]

# ---------------------------
# Modelo Keras dinâmico
# ---------------------------
def build_model(n_hidden, units_list, activation, dropout_rate, lr, input_dim):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_dim,)))
    for i in range(n_hidden):
        model.add(tf.keras.layers.Dense(units_list[i], activation=activation))
        if dropout_rate > 0:
            model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(1, activation="linear"))

    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=opt,
        loss="mse",
        metrics=[
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
        ],
    )
    return model

# ---------------------------
# Função objetivo para Optuna
# ---------------------------
def objective(trial: optuna.Trial):
    # Espaço de busca
    n_hidden = trial.suggest_int("n_hidden", 1, 4)
    units_list = []
    for i in range(4):  # definimos até 4 e usamos apenas n_hidden
        units_list.append(trial.suggest_int(f"units_{i+1}", 16, 512, log=True))
    activation = trial.suggest_categorical("activation", ["relu", "tanh", "sigmoid"])
    dropout_rate = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("learning_rate", 1e-4, 5e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    epochs = 50

    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_rmse", mode="min", patience=20, restore_best_weights=True
    )

    # Um run do MLflow por trial (nested: útil se você quiser um run pai depois)
    with mlflow.start_run(nested=True):
        mlflow.log_params({
            "n_hidden": n_hidden,
            "activation": activation,
            "dropout": dropout_rate,
            "learning_rate": lr,
            "batch_size": batch_size,
            "units_used": str(units_list[:n_hidden]),
        })

        model = build_model(
            n_hidden=n_hidden,
            units_list=units_list,
            activation=activation,
            dropout_rate=dropout_rate,
            lr=lr,
            input_dim=input_dim
        )

        # Registra automaticamente métricas por época (sem logar o modelo a cada época)
        mlflow.keras.autolog(log_models=False)

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[es]
        )

        # Avaliação em validação
        val_metrics = model.evaluate(X_val, y_val, verbose=0, return_dict=True)
        rmse_val = float(val_metrics["rmse"])
        mae_val  = float(val_metrics["mae"])
        mlflow.log_metrics({"val_rmse": rmse_val, "val_mae": mae_val})

        # Loga o modelo deste trial como artefato
        mlflow.keras.log_model(model, artifact_path="trial_model")

    # Minimizar RMSE de validação
    return rmse_val

# ---------------------------
# Otimização com Optuna
# ---------------------------
def run_optuna(n_trials=30):
    study = optuna.create_study(direction="minimize", study_name="diabetes_mlp_optuna_local")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study

# ---------------------------
# Treino final com melhor conjunto
# ---------------------------
def train_final(best_params):
    best_n_hidden = best_params["n_hidden"]
    best_units = [best_params.get(f"units_{i+1}") for i in range(4)]
    final_model = build_model(
        n_hidden=best_n_hidden,
        units_list=best_units,
        activation=best_params["activation"],
        dropout_rate=best_params["dropout"],
        lr=best_params["learning_rate"],
        input_dim=input_dim
    )

    es_final = tf.keras.callbacks.EarlyStopping(
        monitor="val_rmse", mode="min", patience=30, restore_best_weights=True
    )

    with mlflow.start_run(run_name="best_model_local"):
        mlflow.log_params({"stage": "final_train", **best_params})
        mlflow.keras.autolog(log_models=False)

        final_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=300,
            batch_size=best_params["batch_size"],
            verbose=0,
            callbacks=[es_final]
        )

        test_metrics = final_model.evaluate(X_test, y_test, verbose=0, return_dict=True)
        mlflow.log_metrics({
            "test_rmse": float(test_metrics["rmse"]),
            "test_mae":  float(test_metrics["mae"]),
        })

        # Artefatos úteis (scaler + metadados)
        import joblib
        artifacts_dir = os.path.join(here, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)
        joblib.dump(scaler, os.path.join(artifacts_dir, "standard_scaler.joblib"))
        with open(os.path.join(artifacts_dir, "metadata.json"), "w") as f:
            json.dump({"input_dim": int(input_dim)}, f)

        mlflow.log_artifacts(artifacts_dir, artifact_path="preprocessing")
        mlflow.keras.log_model(final_model, artifact_path="best_model")

    return final_model

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    study = run_optuna(n_trials=30)  # ajuste o nº de trials conforme o tempo
    print("Melhor RMSE (val):", study.best_value)
    print("Melhores parâmetros:", study.best_params)
    _ = train_final(study.best_params)
    print("\nConcluído. Para visualizar:\n  1) Abra um terminal na pasta do projeto\n  2) Rode: mlflow ui\n  3) Acesse: http://127.0.0.1:5000\n")
