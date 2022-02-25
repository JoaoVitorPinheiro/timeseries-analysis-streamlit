import pandas as pd
import numpy as np
import math

def MAPE(y_true: pd.Series, y_predicted: pd.Series) -> float:
    """Calcula o Erro Médio Percentual Absoluto (MAPE) multiplicado por 100 para percentual
    Parameters
    ----------
    y_true : pd.Series
        Série de valores observados.
    y_pred : pd.Series
        Série de valores previstos.
    Returns
    ----------
    float
    mape: Mean Absolute Percentage Error (MAPE).
    """
    try:
        residual = np.abs(y_true - y_predicted)
        mape = np.where(y_predicted!=0, residual/y_predicted, np.nan)
        mape = np.where((residual==0) & (y_predicted==0), 0, mape)
        ### mape = np.where(mape > 0.5, 0.1, mape)
        return 100*np.abs(mape)
    except:
        return 0  
    
def MSE(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Computes Mean Squared Error (MSE).
    Parameters
    ----------
    y_true : pd.Series
        Ground truth target series.
    y_pred : pd.Series
        Prediction series.
    Returns
    -------
    float
        Mean Squared Error (MSE).
    """
    try:
        y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
        mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
        mse = ((y_true - y_pred) ** 2)[mask].mean()
        return 0 if np.isnan(mse) else float(mse)
    except:
        return 0  
    
def RMSE(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Computes Root Mean Squared Error (RMSE).
    Parameters
    ----------
    y_true : pd.Series
        Ground truth target series.
    y_pred : pd.Series
        Prediction series.
    Returns
    -------
    float
        Root Mean Squared Error (RMSE).
    """
    y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
    rmse = np.sqrt(MSE(y_true, y_pred))
    return float(rmse)

def MPE(y_true: pd.Series, y_predicted: pd.Series) -> float:
    """Calcula o Erro Médio Percentual(MPE) multiplicado por 100 para percentual
    Parameters
    ----------
    y_true : pd.Series
        Série de valores observados.
    y_pred : pd.Series
        Série de valores previstos.
    Returns
    ----------
    float
    mape: Mean Absolute Percentage Error (MAPE).
    """
    try:
        residual = (y_true - y_predicted)
        mpe = np.where(y_true!=0, residual/y_true, np.nan)
        mpe = np.where((residual==0) & (y_predicted==0), 0, mpe)
        mpe = np.where(mpe > 0.5, 0.1, mpe)
        
        return mpe*100
    except:
        return 0   
    
def RSE(y_true, y_predicted):
    """
    - y_true: Valores Observados
    - y_predicted: Valores Previstos
    """
    y_true = np.array(y_true)
    y_predicted = np.array(y_predicted)
    rss = np.sum(np.square(y_true - y_predicted))
    rse = math.sqrt(rss / (len(y_true) - 2))
    return (y_true - y_predicted)/rse
