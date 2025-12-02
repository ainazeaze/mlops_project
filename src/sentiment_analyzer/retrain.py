import click

@click.command()
@click.option('--model_name', help='Name of the model to use for prediction')
@click.option('--model_version', help='Version of the model to use for prediction')
@click.option('--training_set', help='Path to the training set')
@click.option('--training_set_id', default=None, help='Training set id')
@click.option('--register_updated_model', default=None, help='If defined, save as a new version')
@click.option('--mlflow_url', default='http://localhost:5000', help='Version of the model to use for prediction')
def main(model_name, model_version, status, test_set, mlflow_url):
    print("Ta gueule")
    pass