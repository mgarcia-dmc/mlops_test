import argparse
import mlflow
from mlflow.tracking import MlflowClient
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
from azure.ai.ml.exceptions import ValidationException

def main(args):
    # 1. Conexión explícita al workspace de Azure MLs
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace_name,
    )

    # 2. Conectar el cliente de MLflow al tracking URI de Azure ML
    mlflow_tracking_uri = ml_client.workspaces.get(args.workspace_name).mlflow_tracking_uri
    mlflow_client = MlflowClient(tracking_uri=mlflow_tracking_uri)

    # 3. Obtener la métrica del modelo actual del job
    print(f"Buscando el job con run_id: {args.run_id}")
    current_run = mlflow_client.get_run(args.run_id)
    new_model_auc = current_run.data.metrics.get(args.metric_name)

    if new_model_auc is None:
        raise ValueError(f"Métrica '{args.metric_name}' no encontrada en el job '{args.run_id}'.")
    
    print(f"Nuevo modelo entrenado con AUC: {new_model_auc:.5f}")

    # 4. Encontrar el modelo actual en producción
    prod_model_auc = -1.0
    try:
        # Busca el modelo con la etiqueta "production"
        prod_model = ml_client.models.get(name=args.model_name, label="production")
        prod_run_id = prod_model.properties.get("mlflow.runId")
        if prod_run_id:
            prod_run = mlflow_client.get_run(prod_run_id)
            prod_model_auc = prod_run.data.metrics.get(args.metric_name, -1.0)
        print(f"Modelo en producción encontrado (Versión: {prod_model.version}) con AUC: {prod_model_auc:.5f}")
    except ValidationException:
        print("No se encontró un modelo con la etiqueta 'production'. Esto es normal en la primera ejecución.")

    # 5. Comparar el nuevo modelo con el de producción
    if new_model_auc >= prod_model_auc:
        model_path = (
            f"azureml://subscriptions/{args.subscription_id}/"
            f"resourcegroups/{args.resource_group}/"
            f"workspaces/{args.workspace_name}/"
            f"datastores/workspaceblobstore/"
            f"paths/azureml/{args.run_id}/model_output/"
        )    
        print(f"¡Nuevo modelo es mejor! Registrando y promoviendo a 'production'.\n Ruta del modelo: {model_path}")
    
        # Primero, crea el objeto Modelo sin las etiquetas.
        model_to_register = Model(
            name="insurance-model",
            type=AssetTypes.CUSTOM_MODEL,
            path=model_path,
            tags={"stage": "production"},
            description="Modelo entrenado y evaluado automáticamente"
        )
        
        # Luego, asigna las etiquetas como una propiedad al objeto ya creado.
        model_to_register.labels = {"production": ""}
        # ---------------------------------------------

        # Crea o actualiza el modelo en el registro de Azure ML
        ml_client.models.create_or_update(model_to_register)
        print("Modelo registrado y etiquetado como 'production' exitosamente.")
    else:
        print(f"Nuevo modelo NO es mejor. AUC: {new_model_auc:.5f} < {prod_model_auc:.5f}. Rechazando el despliegue.")
        raise Exception("El nuevo modelo no supera la calidad del modelo en producción.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="insurance_model")
    parser.add_argument("--metric_name", type=str, default="auc")
    parser.add_argument("--subscription_id", type=str, required=True)
    parser.add_argument("--resource_group", type=str, required=True)
    parser.add_argument("--workspace_name", type=str, required=True)
    args = parser.parse_args()
    main(args)