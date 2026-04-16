from pandas import DataFrame
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# 1. Solve the outlier issue in `incentive`
# 2. Keep `wip` only in the sewing department
# 3. Delete `date`
# 4. Encode `day` and `department`
# 5. Handle the high correlation between `no_of_worker` and `smv`

def run(df: DataFrame) -> DataFrame:
    original_rows = len(df)
    incentive_q95 = df['incentive'].quantile(0.95)
    df = df[df['incentive'] <= incentive_q95].copy()
    removed_rows = original_rows - len(df)

    print(f"incentive 95th percentile: {incentive_q95}")
    print(f"Removed rows (incentive > q95): {removed_rows}")
    df['incentive_is_zero'] = (df['incentive'] == 0).astype(int)
    print(df['incentive'].head(), df['incentive_is_zero'].head())

    
    
    return df
