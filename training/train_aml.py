import argparse
import os
import json
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import mlflow
import joblib # Usaremos joblib para un control total

def main(args):
    # Cargar y preparar datos
    df = pd.read_csv(os.path.join(args.data_path, "insurance.csv"))
    X = df.drop(['target', 'id'], axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Cargar parámetros
    with open(os.path.join(args.training_parameters)) as f:
        params = json.load(f)['training']

    # Iniciar un run de MLflow para registrar la métrica
    with mlflow.start_run() as run:
        print(f"MLflow Run ID: {run.info.run_id}")

        # Entrenar modelo
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)

        # Evaluar y registrar la métrica
        y_pred = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        print(f"Registrando métrica AUC: {auc}")
        mlflow.log_metric("auc", auc)

        # --- GUARDADO EXPLÍCITO DEL MODELO ---
        # Guarda el modelo directamente en la ruta de salida que nos da el job.
        # Azure ML se asegura de que esta carpeta exista.
        model_path = os.path.join(args.model_output, "insurance_model.pkl")
        print(f"Guardando el modelo en la ruta de salida: {model_path}")
        joblib.dump(model, model_path)
        print("Modelo guardado exitosamente.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--training_parameters", type=str, required=True)
    parser.add_argument("--model_output", type=str, required=True)
    args = parser.parse_args()
    main(args)