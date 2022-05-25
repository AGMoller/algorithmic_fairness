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
from study_reprojection import data_preprocessing

np.seterr(all="ignore")

warnings.filterwarnings('ignore')


if __name__ == '__main__':

    standardize = True
    standardize_continuous = True
    resample = False
    reproject = False
    lambda_reproject = False
    run_optim_no_fairness = True

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

    if run_optim_no_fairness:
        study = optuna.create_study(direction="maximize",
                                    sampler=sampler,
                                    pruner=optuna.pruners.MedianPruner(
                                        n_startup_trials=2, n_warmup_steps=5,
                                        interval_steps=3),
                                    )

    else:
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

    # Define objective
    objective = Objective(
        train_processed,
        val_processed,
        y_train,
        y_val,
        val_race,
        evaluation_func=equailized_odds,
        run_optim_no_fairness=run_optim_no_fairness)

    # Make a study to optimize the objective.
    study.optimize(
        objective,
        n_trials=100,
        n_jobs=-1,
        show_progress_bar=True)

    if run_optim_no_fairness:
        print(study.best_trial)

    else:
        fig = optuna.visualization.plot_pareto_front(
            study, target_names=["EO", "F1"])
        fig.show()
