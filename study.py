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

np.seterr(all="ignore")

warnings.filterwarnings('ignore')


def data_preprocessing(data: pd.DataFrame, categorical_features:
                       List[str], target_variable: str,
                       non_protected_features: List[str]):

    df = data.copy()

    # protected variable
    race = df["Race"].values

    # target
    target = df[target_variable].values

    df_processed = df[non_protected_features]
    df_processed = pd.get_dummies(
        df_processed,
        prefix=None,
        prefix_sep='_',
        dummy_na=False,
        columns=categorical_features,
        drop_first=True)

    return df_processed, df, target, race


def reproject_features(
        data: pd.DataFrame, protected_cols: List[str], nonprotected_cols: List[str]) -> np.ndarray:
    """
    generate a fair representation of nonprotected columns which are independent from any columns in protected_cols

    data : pd.DataFrame
        dataframe with columns to be projected
    protected_cols : List[str]
        list of protected columns
    nonprotected_cols : List[str]
        list of non-protected columns
    """
    # make a copy of data
    df: pd.DataFrame = data.copy()
    # df is our data
    # extract data about protected columns
    nonprotect: np.ndarray = df[nonprotected_cols].values
    protect: np.ndarray = df[protected_cols].values
    # extract data about nonprotected columns
    debiased_nonprotect: np.ndarray = df[nonprotected_cols].values
    # crease an orthonormal basis
    base_protect: np.ndarray = scipy.linalg.orth(protect)

    # go through all protected attributes and calculate their contribution to
    # the reprojection to the hyperplane
    for j in range(debiased_nonprotect.shape[1]):
        debiased_nonprotect[:,
                            j] -= base_protect @ base_protect.T @ nonprotect[:,
                                                                             j]
    return debiased_nonprotect


def reproject_features_w_regul(
        data: pd.DataFrame, protected_cols: List[str], nonprotected_cols: List[str], lambda_: float) -> np.ndarray:
    """
    generate a fair representation of nonprotected columns which are independent from any columns in protected_cols
    data: a data frame
    protected_cols: list of strings, the protected columns
    nonprotected_col: string, all other data columns
    lambda_: float number between 0 and 1, 0 means totally fair; 1 means same as raw data
    """

    # run the normal reproject_features function
    r: np.ndarray = reproject_features(data, protected_cols, nonprotected_cols)

    # extract data about nonprotected variables
    nonprotect: np.ndarray = data[nonprotected_cols].values
    # standardize columns

    return r + lambda_ * (nonprotect - r)


def equailized_odds(preds: np.ndarray, groups: np.ndarray,
                    test: np.ndarray, verbose: bool = False) -> Union[float, Dict]:
    """
    Calculates the equailized odds of a binary classification problem.
    preds: predictions of the model
    groups: group labels of the test data
    test: test data
    sum_of_differences: if True, the sum of the differences is returned, else the mean of the differences is returned
    verbose: if True, prints the results
    """

    df = pd.DataFrame(list(zip(preds, groups, test)),
                      columns=['preds', 'groups', 'test'])

    # save all results
    all_results = {}

    total_class_difference = 0
    for target in df['test'].unique():
        results = {}
        for group in df['groups'].unique():

            # get the group and amount of corrects in the group
            selection = df.loc[(df['test'] == target) &
                               (df['groups'] == group)]
            corrects = selection.loc[selection['preds'] == 'Yes']

            # if there are no corrects in the group, skip
            if len(corrects) == 0:
                results[group] = 0
                continue

            # get the odds ratio
            score = round(len(corrects) / len(selection), 3)

            # add the score to the results
            results[group] = score

            if verbose:
                print(f'Target [{target}] and group [{group}]: {score} ')

        class_differences = abs(results['White'] - results['Black'])

        if verbose:
            print(results)
            print(f'Class differences for class {group}: {class_differences}')

        # sum up differences or take average
        total_class_difference += class_differences

        all_results[target] = results

    if verbose:
        print(f'Total class difference: {total_class_difference}')

    return total_class_difference, all_results


optuna.logging.set_verbosity(1)


class Objective(object):
    def __init__(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        group_val: np.ndarray,
        evaluation_func: Callable,
        run_optim_no_fairness: bool = False,
    ):
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.group_val = group_val
        self.evaluation_func = evaluation_func,
        self.run_optim_no_fairness = run_optim_no_fairness

    def __call__(self, trial):
        """This method is called by Optuna to compute the objective
        function."""
        # Initialize general hyper parameters

        params = {
            "criterion": trial.suggest_categorical("criterion", ["gini"]),
            "max_depth": trial.suggest_int("max_depth", 20, 50),
            "min_samples_split": trial.suggest_float("min_samples_split", 1e-5, 0.01),
            "min_samples_leaf": trial.suggest_float("min_samples_leaf", 1e-5, 0.01),
        }

        # parameters for fitting a model
        whitebox_model = DecisionTreeClassifier(
            **params, random_state=42).fit(self.X_train, self.y_train)

        preds: np.ndarray = whitebox_model.predict(self.X_val)

        if self.run_optim_no_fairness:
            return metrics.f1_score(self.y_val, preds, labels=[
                                    'Yes'], pos_label="Yes")
        else:
            return self.evaluation_func[0](preds, self.group_val, self.y_val, verbose=False)[
                0], metrics.f1_score(self.y_val, preds, labels=['Yes'], pos_label="Yes")


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
        # "Race",
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

    protected_cols = ['Race']
    nonprotected_cols = [
        f for f in categorical_features +
        continuos_features if f not in protected_cols]

    train_processed, train_original, train_target, train_race = data_preprocessing(
        train, categorical_features=categorical_features,
        target_variable=target_variable,
        non_protected_features=nonprotected_cols)
    val_processed, val_original, val_target, val_race = data_preprocessing(
        val, categorical_features=categorical_features,
        target_variable=target_variable,
        non_protected_features=nonprotected_cols)
    test_processed, test_original, test_target, test_race = data_preprocessing(
        test, categorical_features=categorical_features,
        target_variable=target_variable,
        non_protected_features=nonprotected_cols)

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
        # When we only want to optimize f1 score
        study = optuna.create_study(direction='maximize',
                                    sampler=sampler,
                                    pruner=optuna.pruners.MedianPruner(
                                        n_startup_trials=2, n_warmup_steps=5,
                                        interval_steps=3),)
    else:
        # When we want to optimize fairness and f1 score
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

    if reproject:
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

    else:
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
