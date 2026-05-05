from src.pipeline import LifeExpectancyPipeline

if __name__ == "__main__":
    pipeline = LifeExpectancyPipeline("data/Life Expectancy Data.csv")
    pipeline.run()