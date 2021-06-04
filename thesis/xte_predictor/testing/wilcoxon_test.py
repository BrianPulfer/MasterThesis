import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def main():
    """Computes the Wilcoxon Test between the predictions of the XTE predictor A and XTE predictor B"""

    # If predictions files are not present, exit.
    if not os.path.isfile('./Predictions_A.csv') or not os.path.isfile('./Predictions_B.csv'):
        print("CSV files with XTE predictors predictions missing. Can be obtained by running 'test_xte_predictor.py'.")
        exit()

    # Loading data
    predictions_a = np.array(pd.read_csv('./Predictions_A.csv')).reshape(-1)
    predictions_b = np.array(pd.read_csv('./Predictions_B.csv')).reshape(-1)

    # Running the Wilcoxon Test
    w_statistic, pvalue = wilcoxon(predictions_a, predictions_b)
    print(f"Sum of ranks of differences is: {w_statistic}")
    print(f"P-Value is: {pvalue}")


if __name__ == '__main__':
    main()
