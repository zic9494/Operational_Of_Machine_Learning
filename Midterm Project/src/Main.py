import DataExploration, FeatureEngineering
from pandas import read_csv

try:
    df = read_csv("dataset/garments_worker_productivity.csv")
    DataExploration.run(df)
    df = FeatureEngineering.run(df)
except FileNotFoundError:
    print("Dataset No Found")
except Exception as e:
    print(f"Error :{e}")
