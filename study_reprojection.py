import warnings
from collections import Counter
from typing import Callable, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import scipy
from imblearn.over_sampling import RandomOverSampler
from sklearn import metrics, tree
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sqlalchemy import desc
from tqdm import tqdm

from metrics_eo import equailized_odds
from objective import Objective
from reprojections import reproject_features_w_regul, reproject_features


def data_preprocessing(data: pd.DataFrame, categorical_features:
                       List[str], continuos_features: List[str],
                       target_variable: str):

    df = data.copy()

    # protected variable
    race = df["Race"].values

    # target
    target = df[target_variable].values

    df_processed = df[categorical_features + continuos_features]
    df_processed = pd.get_dummies(
        df_processed,
        prefix=None,
        prefix_sep='_',
        dummy_na=False,
        columns=categorical_features,
        drop_first=True)

    return df_processed, df, target, race


optuna.logging.set_verbosity(1)


if __name__ == '__main__':

    standardize = True
    standardize_continuous = False
    resample = True
    reproject = True
    lambda_reproject = False
    run_optim_no_fairness = False

    if reproject or lambda_reproject:
        assert standardize == True, 'Reprojecting requires standardizing'

    train = pd.read_csv('data/heart_train.csv')
    val = pd.read_csv('data/heart_val.csv')
    test = pd.read_csv('data/heart_test.csv')

    # Select only rows with black and white race
    train = train.loc[train['Race'].isin(['White', 'Black'])]
    val = val.loc[val['Race'].isin(['White', 'Black'])]
    test = test.loc[test['Race'].isin(['White', 'Black'])]

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

    train_processed, train_original, train_target, train_race = data_preprocessing(
        train, categorical_features=categorical_features,
        continuos_features=continuos_features,
        target_variable=target_variable)
    val_processed, val_original, val_target, val_race = data_preprocessing(
        val, categorical_features=categorical_features,
        continuos_features=continuos_features,
        target_variable=target_variable)
    test_processed, test_original, test_target, test_race = data_preprocessing(
        test, categorical_features=categorical_features,
        continuos_features=continuos_features,
        target_variable=target_variable)

    y_train, y_val, y_test = train_target, val_target, test_target

    mean_ = np.mean(train_processed, axis=0)
    std_ = np.std(train_processed, ddof=1, axis=0)

    train_processed = (train_processed - mean_) / std_
    val_processed = (val_processed - mean_) / std_
    test_processed = (test_processed - mean_) / std_

    # Do random oversampling to make class distribution even
    if resample:
        ros = RandomOverSampler(random_state=42)
        train_processed, y_train, = ros.fit_resample(
            train_processed, y_train)

    protected_cols = ['Race_White']
    nonprotected_cols = [
        f for f in train_processed if f not in protected_cols]

    lambda_values = np.linspace(0, 1, 25)
    # Run optimization

    param_grid = {
        'criterion': 'gini',
        'max_depth': 46,
        'min_samples_split': 0.0031163168319514006,
        'min_samples_leaf': 0.002909135794829023}

    results_f1 = {}
    results_eo = {}
    results_class = {
        'TPR_White': [],
        'FPR_White': [],
        'TPR_Black': [],
        'FPR_Black': []}

    for i in tqdm(lambda_values, desc='Lambda'):

        # basic reprojection
        train_processed_r = reproject_features_w_regul(
            train_processed,
            protected_cols=protected_cols,
            nonprotected_cols=nonprotected_cols,
            lambda_=i)

        val_processed_r = reproject_features_w_regul(
            val_processed,
            protected_cols=protected_cols,
            nonprotected_cols=nonprotected_cols,
            lambda_=i)

        test_processed_r = reproject_features_w_regul(
            test_processed,
            protected_cols=protected_cols,
            nonprotected_cols=nonprotected_cols,
            lambda_=i)

        whitebox_model = DecisionTreeClassifier(
            **param_grid, random_state=42).fit(train_processed_r, y_train)

        # Evaluate model
        y_pred = whitebox_model.predict(val_processed_r)

        results_f1[i] = metrics.f1_score(
            y_val, y_pred, labels=['Yes'], pos_label="Yes")
        results_eo[i] = equailized_odds(y_pred, val_race, y_val)[0]
        all_res = equailized_odds(y_pred, val_race, y_val)[1]
        results_class["TPR_White"].append(all_res["Yes"]['White'])
        results_class["FPR_White"].append(all_res["No"]['White'])
        results_class["TPR_Black"].append(all_res["Yes"]['Black'])
        results_class["FPR_Black"].append(all_res["No"]['Black'])
    a = 1
