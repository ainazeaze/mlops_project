import mlflow
from mlflow.tracking import MlflowClient

class ModelManager:
    def __init__(self, model_name, model_version, mlflow_url):
        self.model_name = model_name
        self.model_version = model_version
        self.mlflow_url = mlflow_url
        mlflow.set_tracking_uri(self.mlflow_url)
        if model_name is None or model_version is None:
            raise Exception("TEST_MODEL_NAME and TEST_MODEL_VERSION is not defined")
        model_uri = f"models:/{model_name}/{model_version}"
        self._model = mlflow.sklearn.load_model(model_uri)


    def get_model(self):
        return self._model
    
    def promote(self, status):
        stage_map = {"Staging" : "Production", "None": "Staging"}            
        client = MlflowClient()
        model_version = client.get_model_version(name=self.model_name, version=self.model_version)
        if status != stage_map[model_version.current_stage]:
            raise Exception(f"Cannot go from {model_version.current_stage} to {status}")
        
        client.transition_model_version_stage(
            name=self.model_name,
            version=self.model_version,
            stage=status,
        )

    def retrain(self, training_set, training_set_id, register_updated_model):
        pass

    