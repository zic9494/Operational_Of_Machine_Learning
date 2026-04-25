import DataExploration, FeatureEngineering, TrainingModel
from joblib import dump
from pandas import read_csv
from pathlib import Path
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "garments_worker_productivity.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "linear_regression_model.joblib"
ONNX_MODEL_PATH = PROJECT_ROOT / "models" / "linear_regression_model.onnx"

try:
    df = read_csv(DATASET_PATH)
    DataExploration.run(df)
    X_train, X_test, y_train, y_test = FeatureEngineering.run(df)
    model = TrainingModel.run_linear_regression(X_train, X_test, y_train, y_test)
    
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(model, MODEL_PATH)
    print(f"\nModel exported to: {MODEL_PATH}")

    initial_type = [("float_input", FloatTensorType([None, X_train.shape[1]]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    ONNX_MODEL_PATH.write_bytes(onnx_model.SerializeToString())
    print(f"ONNX model exported to: {ONNX_MODEL_PATH}")
except FileNotFoundError:
    print("Dataset No Found")
except Exception as e:
    print(f"Error :{e}")
