import numpy as np
import optuna
from typing import Callable, Dict, List, Tuple, Union
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


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
