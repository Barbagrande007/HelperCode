# feature_importance.py
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt


def feature_importance(x_feat, y_feat, seed, regression=True):
    """
    Creates a horizontal bar plot with features sorted on importance
    -----------------------------------------------------------------
    :param x_feat: DataFrame
        Containing the feature data X
    :param y_feat: Series
        Containing the labels y
    :param seed: int
        Global parameter for random seed
    :param regression: bool
        If true it will use regression, when false classification
    :return:
        A graph with features (X) sorted by importance
    """

    if regression:
        rf = RandomForestRegressor(random_state=seed)
    else:
        rf = RandomForestClassifier(random_state=seed)

    rf.fit(x_feat, y_feat)
    imp = rf.feature_importances_
    df = pd.DataFrame(imp, index=x_feat.columns, columns=["Importance"])

    fig, ax = plt.subplots(figsize=(10, 6))

    sorted_idx = imp.argsort()
    ax.barh(df.index[sorted_idx], df['Importance'][sorted_idx], height=0.8, facecolor='grey', alpha=0.8, edgecolor='k')
    ax.set_xlabel('Importance score')
    ax.set_title('Permutation feature importance')
    plt.gca().invert_yaxis()
    fig.tight_layout()
    plt.show()
