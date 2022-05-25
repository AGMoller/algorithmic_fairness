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

from metrics_eo import equailized_odds
from objective import Objective
from reprojections import reproject_features_w_regul, reproject_features

np.seterr(all="ignore")

warnings.filterwarnings('ignore')


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

    if standardize:
        if standardize_continuous:
            # Scale continuous variables
            mean_ = np.mean(train_processed[continuos_features], axis=0)
            std_ = np.std(train_processed[continuos_features], ddof=1, axis=0)

            train_processed = (
                train_processed[continuos_features] - mean_) / std_
            val_processed = (val_processed[continuos_features] - mean_) / std_
            test_processed = (
                test_processed[continuos_features] - mean_) / std_

        else:
            # Standardize all variables
            mean_ = np.mean(train_processed, axis=0)
            std_ = np.std(train_processed, ddof=1, axis=0)

            train_processed = (train_processed - mean_) / std_
            val_processed = (val_processed - mean_) / std_
            test_processed = (test_processed - mean_) / std_

    sampler = optuna.samplers.TPESampler()

    study = optuna.create_study(directions=["minimize", "maximize"],
                                sampler=sampler,
                                pruner=optuna.pruners.MedianPruner(
                                    n_startup_trials=2, n_warmup_steps=5,
                                    interval_steps=3),
                                )

    # Do random oversampling to make class distribution even
    if resample:
        ros = RandomOverSampler(random_state=42)
        train_processed, y_train, = ros.fit_resample(
            train_processed, y_train)

    protected_cols = ['Race_White']
    nonprotected_cols = [
        f for f in train_processed if f not in protected_cols]

    if lambda_reproject:

        lambda_values = np.linspace(0, 1, 10)
        # Run optimization

        for i in lambda_values:

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

            # Define objective
            objective = Objective(
                train_processed_r,
                val_processed_r,
                y_train,
                y_val,
                val_race,
                equailized_odds,
                run_optim_no_fairness=run_optim_no_fairness)

            study.optimize(objective, n_trials=100,
                           verbose=False,
                           n_jobs=-1,
                           )

            # plot
            fig = optuna.visualization.plot_pareto_front(
                study, target_names=["EO", "F1"])

            fig.update_layout(
                title=f"Pareto Front Plot, lambda: {i}")
            fig.show()

    else:

        print('Hello now im here!')
        # basic reprojection
        train_processed_r = reproject_features(
            train_processed,
            protected_cols=protected_cols,
            nonprotected_cols=nonprotected_cols)

        val_processed_r = reproject_features(
            val_processed,
            protected_cols=protected_cols,
            nonprotected_cols=nonprotected_cols)

        test_processed_r = reproject_features(
            test_processed,
            protected_cols=protected_cols,
            nonprotected_cols=nonprotected_cols)

        # Define objective
        objective = Objective(
            train_processed_r,
            val_processed_r,
            y_train,
            y_val,
            val_race,
            equailized_odds,
            run_optim_no_fairness=run_optim_no_fairness)

        study.optimize(objective, n_trials=100,
                       verbose=False,
                       n_jobs=-1,)

        fig = optuna.visualization.plot_pareto_front(
            study, target_names=["EO", "F1"])
        fig.show()
