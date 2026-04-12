import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from data_fetch import DataFetch


class DataProcessor:
    def __init__(self):
        self.df = self.fetch_data()

    # pobranie danych
    def fetch_data(self):
        self.df = DataFetch().fetch_dataset().data.copy()

    def show_basic_info(self):
        print("\nInfo:")
        print(self.df.info())


if __name__ == '__main__':
    pass