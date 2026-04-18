from pandas import DataFrame, Series, get_dummies, isna
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
# 1. Solve the outlier issue in `incentive`
# 2. Keep `wip` only in the sewing department
# 3. Delete `date`
# 4. Encode `day` and `department`
# 5. Handle the high correlation between `no_of_worker` and `smv`

def run(
    df: DataFrame,
    target_col: str = "actual_productivity",
    scaler_type: str = "standard",
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[DataFrame, DataFrame, Series, Series]:
    df = df.copy()

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe.")

    # delete outliner in `incentive`
    incentive_q95 = df["incentive"].quantile(0.95)
    df = df[df["incentive"] <= incentive_q95].copy()

    # notice that the data has two type one is work hard to get incentive and the other is not to get incentive
    df["incentive_is_zero"] = (df["incentive"] == 0).astype(int)
    department_normalized = df["department"].astype(str).str.strip().str.lower()
    
    # mark the deportment is "sewing"
    df["wip_applicable"] = (department_normalized == "sewing").astype(int)
    df["is_sewing"] = df["wip_applicable"]
    if "wip" in df.columns and "department" in df.columns:
        # find out every "wip" is nan
        wip_is_missing = df["wip"].isna()
        df["wip_missing"] = (wip_is_missing & (df["wip_applicable"] == 1)).astype(int)

        # find median of deportment is "sewing" 
        sewing_wip_median = df.loc[df["wip_applicable"] == 1, "wip"].median()
        if isna(sewing_wip_median):
            sewing_wip_median = 0.0

        # if deparment is "sewing" and data is missing fill it up with median
        df["wip_filled"] = df["wip"]
        df.loc[(df["wip_applicable"] == 1) & wip_is_missing, "wip_filled"] = sewing_wip_median
        df.loc[df["wip_applicable"] == 0, "wip_filled"] = 0.0
        df = df.drop(columns=["wip"])

    
    if "date" in df.columns:
        df = df.drop(columns=["date"])

    # one hot encoding
    categorical_cols = [col for col in ["day", "department"] if col in df.columns]
    df = get_dummies(df, columns=categorical_cols, drop_first=False)

    # split into X and y for linear regression preprocessing
    y = df[target_col]
    X = df.drop(columns=[target_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    # filte out logic columns and number columns
    numeric_cols = list(X_train.select_dtypes(include=["number"]).columns)
    binary_cols = []
    for col in numeric_cols:
        unique_values = set(X_train[col].dropna().unique().tolist())
        if unique_values.issubset({0, 1}):
            binary_cols.append(col)
    scale_cols = [col for col in numeric_cols if col not in binary_cols]
    
    # standardization 
    if len(scale_cols) > 0:
        if scaler_type == "robust":
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()

        X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
        X_test[scale_cols] = scaler.transform(X_test[scale_cols])

    return X_train, X_test, y_train, y_test
