from pandas import DataFrame, Series
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

def run_linear_regression(
    X_train: DataFrame, 
    X_test: DataFrame, 
    y_train:Series, 
    y_test:Series
) -> LinearRegression:
    
    Regressor = LinearRegression()
    Regressor.fit(X_train, y_train)
    y_Pred = Regressor.predict(X_test)

    print("\nModel result:")
    print('R Square:%f'%Regressor.score(X_test,y_test))
    print('MAE:%f'%mean_absolute_error(y_true = y_test, y_pred=y_Pred))
    mse = mean_squared_error(y_true=y_test, y_pred=y_Pred)
    rmse = mse ** 0.5
    print('MSE:%f'%mse)
    print('RMSE:%f'%rmse)
    return Regressor
