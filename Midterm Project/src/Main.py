import DataExploration, FeatureEngineering, TrainingModel
from pandas import read_csv

try:
    df = read_csv("dataset/garments_worker_productivity.csv")
    DataExploration.run(df)
    X_train, X_test, y_train, y_test = FeatureEngineering.run(df)
    model = TrainingModel.run_linear_regression(X_train, X_test, y_train, y_test)
except FileNotFoundError:
    print("Dataset No Found")
except Exception as e:
    print(f"Error :{e}")
