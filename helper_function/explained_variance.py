# explained_variance.py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def explained_variance(data, labels, scale=True):
    """
    Creates a graph features and their explained variance
    :param data: dataframe
        X_features
    :param labels: series
        y
    :param scale: boolean
        data will be scaled if True
    :return: Explained variance graph
    """
    if scale:
        # Scale data
        std_scaler = StandardScaler()
        data = std_scaler.fit_transform(data)

    # Instantiate PCA
    pca = PCA()

    # Fit data
    pca.fit_transform(data, labels)

    # Determine explained variance using explained_variance_ration_ attribute
    exp_var_pca = pca.explained_variance_ratio_

    # Cumulative sum of eigenvalues
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)

    # Create the visualization plot
    plt.bar(range(1, len(exp_var_pca)+1), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(1, len(cum_sum_eigenvalues)+1), cum_sum_eigenvalues, where='mid',
             label='Cumulative explained variance')
    plt.axhline(0.85, color="red", linestyle="--", label="85%")
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
