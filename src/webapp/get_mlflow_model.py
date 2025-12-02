import click
import mlflow
import mlflow.sklearn

@click.command()
@click.option("--mlflow_server_uri",  default='http://localhost:5000')
@click.option("--model_name")
@click.option("--model_version")
@click.option("--target_path")

def main(mlflow_server_uri, model_name, model_version, target_path):
    mlflow.set_tracking_uri(mlflow_server_uri)
    model = mlflow.sklearn.load_model(model_uri=f'models:/{model_name}/{model_version}')
    mlflow.sklearn.save_model(model, target_path)

if __name__ == "__main__":
    main()