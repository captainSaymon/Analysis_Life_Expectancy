import kagglehub
from kagglehub import KaggleDatasetAdapter
import shutil
import os

class DataFetch:
    def __init__(self):
        self.data = None

    def fetch_dataset(self):
        self.data = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "kumarajarshi/life-expectancy-who",
            "Life Expectancy Data.csv"
            )
        print("Dane pobrane do cache")
        return self
    
    def clear_dataset(self):
        cache_dir = os.path.expanduser("~/.cache/kagglehub")

        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print("Cache wyczyszczony")
        else:
            print("Dane w cache nie istnieja")
