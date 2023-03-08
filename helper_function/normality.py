# normality.py
import matplotlib.pyplot as plt
import pylab
import scipy.stats as stats
import seaborn as sns


def normality(data, feature):
    """
    Checks for a normal distribution
    :param data: dataframe
    :param feature: selected feature
    :return: three graphs showing: Outliers, KDE, Distribution
    """

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    sns.boxplot(data=data, y=feature)
    plt.title("Outliers")
    plt.subplot(1, 3, 2)
    sns.kdeplot(data[feature])
    plt.title("KDE plot")
    plt.subplot(1, 3, 3)
    stats.probplot(data[feature], plot=pylab)
    plt.title("Distribution per quantiles")
    plt.suptitle(f"{feature}")
    plt.tight_layout()
    plt.show()
