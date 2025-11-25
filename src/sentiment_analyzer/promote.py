import click
import subprocess
import pkg_resources
import os
from sentiment_analyzer.model_manager import ModelManager

@click.command()
@click.option('--model_name', help='Name of the model to use for prediction')
@click.option('--model_version', help='Version of the model to use for prediction')
@click.option('--status', help='Status for which the model is to be promoted to')
@click.option('--test_set', default=None, help='Test dataset')
@click.option('--mlflow_url', default='http://localhost:5000', help='Version of the model to use for prediction')
def main(model_name, model_version, status, test_set, mlflow_url):
    os.environ["TEST_MODEL_NAME"] = model_name
    os.environ["TEST_MODEL_VERSION"] = model_version
    os.environ["TEST_TEST_SET"] = test_set
    
    if status == "Production":
        test_result = subprocess.run(
        ["pytest", pkg_resources.resource_filename('sentiment_analyzer', "./tests")], capture_output=False)
        if test_result.returncode == 0:
            model_manager = ModelManager(model_name=model_name, model_version=model_version, mlflow_url=mlflow_url)
            model_manager.promote(status)
            return
    model_manager = ModelManager(model_name=model_name, model_version=model_version, mlflow_url=mlflow_url)
    model_manager.promote(status)
