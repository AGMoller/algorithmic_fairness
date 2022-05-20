import optuna
from sklearn import metrics
from typing import Callable, List
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv('data/heart_train.csv')
val = pd.read_csv('data/heart_val.csv')
test = pd.read_csv('data/heart_test.csv')

# Preprocessing data

categorical_features = [
    "Smoking",
    "AlcoholDrinking",
    "Stroke",
    "DiffWalking",
    "Sex",
    "AgeCategory",
    "Race",
    "Diabetic",
    "PhysicalActivity",
    "GenHealth",
    "Asthma",
    "KidneyDisease",
    "SkinCancer"
]

continuos_features = [
    "BMI",
    "PhysicalHealth",
    "MentalHealth"
]

target_variable = "HeartDisease"


def data_preprocessing(data: pd.DataFrame, categorical_features: List[str], continuous_features: List[str], target_variable: str):

    df = data.copy()

    # protected variables
    sex = df["Sex"].values
    age = df["AgeCategory"].values
    race = df["Race"].values

    # target
    target = df[target_variable].values

    df_processed = df[categorical_features + continuous_features]
    df_processed = pd.get_dummies(df_processed, prefix=None, prefix_sep='_',
                                  dummy_na=False, columns=categorical_features, drop_first=True)

    return df_processed, df, target, sex, age, race


#df_processed, df_original, target, sex, age, race = data_preprocessing(df, categorical_features=categorical_features, continuous_features=continuos_features, target_variable=target_variable)
train_processed, train_original, train_target, train_sex, train_age, train_race = data_preprocessing(
    train, categorical_features=categorical_features, continuous_features=continuos_features, target_variable=target_variable)
val_processed, val_original, val_target, val_sex, val_age, val_race = data_preprocessing(
    val, categorical_features=categorical_features, continuous_features=continuos_features, target_variable=target_variable)


y_train, y_val = train_target, val_target


optuna.logging.set_verbosity(1)


class Objective(object):
    def __init__(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        evaluation_func: Callable,
    ):
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.evaluation_func = evaluation_func

    def __call__(self, trial) -> float:
        """This method is called by Optuna to compute the objective
        function."""
        # Initialize general hyper parameters

        params = {
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "max_depth": trial.suggest_int("max_depth", 5, 15, step=2),
            "min_samples_split": trial.suggest_loguniform("min_samples_split", 1e-3, 0.01),
            "min_samples_leaf": trial.suggest_loguniform("min_samples_leaf", 1e-3, 0.01),
        }

        # parameters for fitting a model
        whitebox_model = DecisionTreeClassifier(
            **params, random_state=42).fit(self.X_train, self.y_train)

        preds: np.ndarray = whitebox_model.predict(self.X_val)

        return self.evaluation_func(preds, self.y_val, average="macro")


# Run optimization
study = optuna.create_study(direction="maximize")

# Scaling
# Scale continuous variables
scaler = ColumnTransformer(
    [('scaler', StandardScaler(), continuos_features)], remainder='passthrough')
X_train = scaler.fit_transform(train_processed)
X_val = scaler.transform(val_processed)

# Define objective
objective = Objective(X_train, X_val, y_train, y_val, metrics.f1_score)

# Make a study to optimize the objective.
study.optimize(objective, n_trials=100, n_jobs=-1, show_progress_bar=True)
