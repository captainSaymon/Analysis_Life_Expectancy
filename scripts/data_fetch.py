import kagglehub
from kagglehub import KaggleDatasetAdapter

class DataFetch:
    def __init__(self):
        self.data = None

    def run(self):
        self.data = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "kumarajarshi/life-expectancy-who",
            "Life Expectancy Data.csv"
            )
        return self

    def printNumberRecords(self, n):
        print(f"\nFirst {n} records: ", self.data.head(n))

if __name__ == '__main__':
    df = DataFetch()
    df.run()
    df.printNumberRecords(10)
