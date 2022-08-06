import os
from math import sqrt
import joblib
import pandas as pd
from google.cloud import storage
from sklearn.metrics import mean_absolute_error, mean_squared_error

MODEL_NAME = 'taxifare'
PATH_TO_LOCAL_MODEL = 'model.joblib'
AWS_BUCKET_TEST_PATH = "s3://wagon-public-datasets/taxi-fare-test.csv"
BUCKET_NAME = "wagon-data-903-gurung"  # :warning: replace with your BUCKET NAME
params = dict(nrows=10000,
              upload=True,
              local=False,  # set to False to get data from GCP (Storage or BigQuery)
              gridsearch=False,
              optimize=True,
              estimator="xgboost",
              mlflow=True,  # set to True to log params to mlflow
              experiment_name=experiment,
              pipeline_memory=None, # None if no caching and True if caching expected
              distance_type="manhattan",
              feateng=["distance_to_center", "direction", "distance", "time_features", "geohash"],
              n_jobs=-1) # Try with njobs=1 and njobs = -1
def get_test_data(nrows, data="local"):
    """method to get the test data (or a portion of it) from google cloud bucket
    To predict we can either obtain predictions from train data or from test data"""
    # Add Client() here
    path = "data/test.csv"  # :warning: to test from actual KAGGLE test set for submission
    if data == "local":
        df = pd.read_csv(path)
    elif data == "full":
        df = pd.read_csv(AWS_BUCKET_TEST_PATH)
    else:
        df = pd.read_csv(AWS_BUCKET_TEST_PATH, nrows=nrows)
    return df

def download_model(model_directory="PipelineTest", bucket=BUCKET_NAME, rm=True):
    client = storage.Client().bucket(bucket)
    storage_location = 'models/{}/versions/{}/{}'.format(
        MODEL_NAME,
        model_directory,
        'model.joblib')
    blob = client.blob(storage_location)
    blob.download_to_filename('model.joblib')
    print("=> pipeline downloaded from storage")
    model = joblib.load('model.joblib')
    if rm:
        os.remove('model.joblib')
    return model

def get_model(path_to_joblib):
    pipeline = joblib.load(path_to_joblib)
    return pipeline

def evaluate_model(y, y_pred):
    MAE = round(mean_absolute_error(y, y_pred), 2)
    RMSE = round(sqrt(mean_squared_error(y, y_pred)), 2)
    res = {'MAE': MAE, 'RMSE': RMSE}
    return res

def generate_submission_csv(nrows, kaggle_upload=False):
    df_test = get_test_data(nrows)
    pipeline = joblib.load(PATH_TO_LOCAL_MODEL)
    if "best_estimator_" in dir(pipeline):
        y_pred = pipeline.best_estimator_.predict(df_test)
    else:
        y_pred = pipeline.predict(df_test)
    df_test["fare_amount"] = y_pred
    df_sample = df_test[["key", "fare_amount"]]
    name = f"predictions_test_ex.csv"
    df_sample.to_csv(name, index=False)
    print("prediction saved under kaggle format")
    # Set kaggle_upload to False unless you install kaggle cli
    if kaggle_upload:
        kaggle_message_submission = name[:-4]
        command = f'kaggle competitions submit -c new-york-city-taxi-fare-prediction -f {name} -m "{kaggle_message_submission}"'
        os.system(command)
def df_optimized(df, verbose=True, **kwargs):
    """
    Reduces size of dataframe by downcasting numeircal columns
    :param df: input dataframe
    :param verbose: print size reduction if set to True
    :param kwargs:
    :return: df optimized
    """
    in_size = df.memory_usage(index=True).sum()
    # Optimized size here
    for type in ["float", "integer"]:
        l_cols = list(df.select_dtypes(include=type))
        for col in l_cols:
            df[col] = pd.to_numeric(df[col], downcast=type)
            if type == "float":
                df[col] = pd.to_numeric(df[col], downcast="integer")
    out_size = df.memory_usage(index=True).sum()
    ratio = (1 - round(out_size / in_size, 2)) * 100
    GB = out_size / 1000000000
    if verbose:
        print("optimized size by {} % | {} GB".format(ratio, GB))
    return df


if __name__ == '__main__':
    # :warning: in order to push a submission to kaggle you need to use the WHOLE dataset
    nrows = 100
    generate_submission_csv(nrows, kaggle_upload=False)
