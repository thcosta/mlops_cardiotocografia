# -*- coding: utf-8 -*-
import os
import random
import mlflow
import dagshub
import numpy as np
import random as python_random
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, InputLayer
from keras.utils import to_categorical

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def reset_seeds():
  """Resetting seeds for reproducibility"""
  os.environ['PYTHONHASHSEED']=str(42)
  tf.random.set_seed(42)
  np.random.seed(42)
  random.seed(42)

def config_mlflow():
  """Configuring MLflow and DagsHub integration"""

  dagshub.init(repo_owner='thcosta', repo_name='mlops_cardiotocografia', mlflow=True)

  mlflow.set_tracking_uri('https://dagshub.com/thcosta/mlops_cardiotocografia.mlflow')

  mlflow.tensorflow.autolog(log_models=True,
                            log_input_examples=True,
                            log_model_signatures=True)


def read_data() -> (pd.DataFrame, pd.Series): # type: ignore
  """Read dataset and 
    Returns:
        X (pandas.DataFrame): Features dataframe
        y (pandas.Series): Labels series
  """
  data = pd.read_csv('dataset/fetal_health_reduced.csv')
  X = data.drop(["fetal_health"], axis=1)
  y = data["fetal_health"]
  return X, y


def process_data(X: pd.DataFrame, y: pd.Series) -> (pd.DataFrame,
                                                    pd.DataFrame, 
                                                    pd.Series, 
                                                    pd.Series):
    """ Processing data and splitting into train and test sets 
    Returns:
        X_train (pandas.DataFrame): Training features
        X_test (pandas.DataFrame): Testing features
        y_train (pandas.Series): Training labels
        y_test (pandas.Series): Testing labels
    """
    columns_names = list(X.columns)
    scaler = preprocessing.StandardScaler()
    X_df = scaler.fit_transform(X)
    X_df = pd.DataFrame(X_df, columns=columns_names)

    X_train, X_test, y_train, y_test = train_test_split(X_df,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=42)

    y_train = y_train -1
    y_test = y_test - 1
    return X_train, X_test, y_train, y_test


def create_model(X_train: pd.DataFrame) -> Sequential:
  """ Creating and compiling model
  Returns:
      model (Sequential): Compiled Keras Sequential model
  """
  reset_seeds()
  model = Sequential()
  model.add(InputLayer(shape=(X_train.shape[1], )))
  model.add(Dense(10, activation='relu'))
  model.add(Dense(10, activation='relu'))
  model.add(Dense(3, activation='softmax'))

  model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
  return model


def train_model(model: Sequential, 
                X_train: pd.DataFrame, 
                y_train: pd.Series,
                is_train: bool = True) -> None:
  """Function to train the model with MLflow tracking
  Parameters:
      model (Sequential): Keras Sequential model to be trained
      X_train (pandas.DataFrame): Training features
      y_train (pandas.Series): Training labels
      is_train (bool): Flag to indicate if it's training phase  
  Returns:
      None
  """
  with mlflow.start_run(run_name='experiment_mlops_cardiotocografia') as run:
    model.fit(X_train,
              y_train,
              epochs=50,
              validation_split=0.2,
              verbose=3)

if __name__ == "__main__":
    config_mlflow()
    X, y = read_data()
    X_train, X_test, y_train, y_test = process_data(X, y)
    model = create_model(X_train)
    train_model(model, X_train, y_train)