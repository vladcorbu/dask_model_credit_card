import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score,precision_score, recall_score
from lightgbm import LGBMClassifier 
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter
import joblib
from dask.distributed import Client
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split
import pickle
import boto3
import os
import logging
from botocore.exceptions import ClientError


def upload_file(file_name, bucket, object_name=None):

    if object_name is None:
        object_name = os.path.basename(file_name)

    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def ensemble_model_pickle(X_train, Y_train):
    class_weight={0:0.087, 1:1}
    model = LGBMClassifier(n_estimators=200,
    is_unbalance=True,learning_rate=0.1, 
    class_weight=class_weight,
    num_leaves=200,
    random_state=42,
    n_jobs=-1)
    model.fit(X_train, Y_train)
    filename = 'model.sav'
    pickle.dump(model, open(filename, 'wb'))


def execute_model():


    df = dd.read_csv('creditcard.csv', assume_missing=True)
    print(df.info())

    

    features = df.drop(['Class'], axis = 1)
    target = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(features, 
    target, 
    train_size=0.70, 
    test_size=0.30, 
    random_state=1, 
    shuffle=True)

    

    sm = SMOTE(sampling_strategy='minority')
    X_train, y_train = sm.fit_resample(X_train, y_train)


    ensemble_model_pickle(X_train, y_train)

    loaded_model = pickle.load(open('model.sav', 'rb'))

    upload_file('model.sav', 'glue-bucket-processs')

    predicted = loaded_model.predict(X_test)
    print("Recall Score: {}".format(recall_score(y_test, predicted)))
    print("Accuracy Score: {}".format(accuracy_score(y_test, predicted)))
    print("F1 Score: {}".format(f1_score(y_test, predicted)))
    print("Precision Score: {}".format(precision_score(y_test, predicted)))
    

if __name__ == '__main__':
    client = Client()
    with joblib.parallel_backend("dask"):
        execute_model()