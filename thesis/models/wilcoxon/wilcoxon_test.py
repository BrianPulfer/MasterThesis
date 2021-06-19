import os
from os.path import join as join
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def main():
    """Computes the Wilcoxon Test between the models (online and offline)"""

    online_dir = join(os.getcwd(), "online_data")
    offline_dir = join(os.getcwd(), "offline_data")

    a = np.array(pd.read_csv(join(offline_dir, 'MSE_SIM_default.txt'))).reshape(-1)
    b = np.array(pd.read_csv(join(offline_dir, 'MSE_SIM_epoch.txt'))).reshape(-1)

    w_statistic, pvalue = wilcoxon(a, b)
    print(w_statistic, pvalue)


if __name__ == '__main__':
    main()
