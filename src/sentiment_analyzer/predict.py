import click
from sentiment_analyzer.model_manager import ModelManager
import pandas as pd

@click.command()
@click.option('--input_file', default= None, help='Input file path')
@click.option('--output_file', default= None, help='Output file path')
@click.option('--text', default=None, help='Text to use for prediction if input_file is not given')
@click.option('--model_name', help='Name of the model to use for prediction')
@click.option('--model_version', help='Version of the model to use for prediction')
@click.option('--mlflow_url', default='http://localhost:5000', help='Version of the model to use for prediction')
def main(input_file, output_file, text, model_name, model_version, mlflow_url):
    model_manager = ModelManager(model_name=model_name, model_version=model_version, mlflow_url=mlflow_url)
    model = model_manager.get_model()
    if input_file is not None:
        if output_file is None:
            raise Exception("Error output_file must be defined")
            
        data = pd.read_csv(input_file)
        data["polarity"] = model.predict(data["review"])
        data.to_csv(output_file, index=False)
    elif text is not None:
        data = pd.DataFrame()
        data["review"] = [text]
        print(f"Polarity prediction for <{text}> : {model.predict([text])[0]}")
    else:
        raise Exception("Error on of input_file or text must be given")
    
