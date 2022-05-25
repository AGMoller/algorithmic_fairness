from typing import Dict, Union

import numpy as np
import pandas as pd


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
