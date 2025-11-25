import mlflow.sklearn
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_regression
import os
import numpy as np
import pandas as pd
import pytest

remote_server_uri = "http://localhost:5000/"

def load_model():
    mlflow.set_tracking_uri(remote_server_uri)
    model_name = os.environ.get('TEST_MODEL_NAME', None)
    model_version = os.environ.get('TEST_MODEL_VERSION',None)
    if model_name is None or model_version is None:
        raise Exception("TEST_MODEL_NAME and TEST_MODEL_VERSION is not defined")
    model_uri = f"models:/{model_name}/{model_version}"
    return mlflow.sklearn.load_model(model_uri)


def test_model_output_type_with_simple_input():
    # load the model from Mlflow registry
    model = load_model()
    # infer using the model
    input = "Très bon film !!"
    prediction =  model.predict([input])
    assert type(prediction[0]) is np.int64


def test_model_work_with_unusual_input():
    # load the model from Mlflow registry
    model = load_model()

    # infer using the model
    input = "Ce film est nul à chier, &é(-è_çà)" # input with special character
    prediction =  model.predict([input])
    assert type(prediction[0]) is np.int64


def test_model_work_with_empty_input():
    with pytest.raises(ValueError):
        model = load_model()
        model.predict([])

def test_model_prediction_with_obvious_input():
    # load the model from Mlflow registry
    model = load_model()

    # infer using the model
    input = "C'était très bien !" # polarity should be 1 
    prediction =  model.predict([input])
    assert prediction[0]==1


@pytest.mark.skipif( os.environ.get('TEST_TEST_SET', None) is None, reason='TEST_TEST_SET is not provided')
def test_model_accuracy():
    threshold = 0.8
    # load the model from Mlflow registry
    model = load_model()
    test_set = os.environ.get('TEST_TEST_SET', None)
    if test_set is None:
        return 
    df_test = pd.read_csv(test_set)
    Y_test = df_test["polarity"].to_numpy()

    # infer using the model
    prediction =  model.predict(df_test["review"])
    acc_score = accuracy_score(prediction, Y_test)

    assert acc_score > threshold

@pytest.mark.skipif(os.environ.get('TEST_TEST_SET', None) is None or os.environ.get('TEST_BASELINE_MODEL', None) is None
                    or os.environ.get('TEST_BASELINE_VERSION', None) is None   , reason='TEST_TEST_SET or TEST_BASELINE_MODEL or'
                    'TEST_BASELINE_VERSION is not provided')
def test_model_better_than_baseline():
    model = load_model()

    test_set = os.environ.get('TEST_TEST_SET', None)
    df_test = pd.read_csv(test_set)
    Y_test = df_test["polarity"].to_numpy()
    # infer using the model
    prediction =  model.predict(df_test["review"])
    model_acc_score = accuracy_score(prediction, Y_test)

    model_name = os.environ.get('TEST_BASELINE_MODEL', None)
    model_version = os.environ.get('TEST_BASELINE_VERSION',None)
    if model_name is None or model_version is None:
        raise Exception("TEST_MODEL_NAME and TEST_MODEL_VERSION is not defined")
    model_uri = f"models:/{model_name}/{model_version}"
    baseline_model = mlflow.sklearn.load_model(model_uri)
    prediction =  baseline_model.predict(df_test["review"])
    baseline_model_acc_score = accuracy_score(prediction, Y_test)

    assert model_acc_score >= baseline_model_acc_score


