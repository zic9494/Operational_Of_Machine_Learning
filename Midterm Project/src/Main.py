import DataExploration
from pandas import read_csv

try:
    df = read_csv("dataset/garments_worker_productivity.csv")
    print(df.head())
    DataExploration.run(df)
except FileNotFoundError:
    print("Dataset No Found")
except Exception as e:
    print(f"Error :{e}")
