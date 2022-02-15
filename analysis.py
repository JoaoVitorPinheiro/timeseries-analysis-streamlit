
from scipy.stats import shapiro

import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from typing import List, Any, Dict, Tuple

def preprocess_dataframe(data: pd.DataFrame,
                         data_group: str,
                         time_col: str,
                         y_true: str,
                         y_predicted: str,
                         ) -> pd.DataFrame:
    
    data[time_col] = pd.to_datetime(data[time_col],format = '%Y-%m-%d')
    data[time_col] = data[time_col].dt.date
    return data 

def read_from_csv(data_file: pd.DataFrame,
                  time_col: str,
                  y_true: str,
                  y_predicted: str
                ) -> pd.DataFrame:
    
    return

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
    
        mape = np.where(y_true!=0, residual/y_true, np.nan)
        
        mape = np.where((residual==0) & (y_predicted==0), 0, mape)
        
        mape = np.where(mape > 0.5, 0.1, mape)
        
        return mape*100
    except:
        return 0   

def RSE(y_true, y_predicted):
    """
    - y_true: Actual values
    - y_predicted: Predicted values
    """
    y_true = np.array(y_true)
    y_predicted = np.array(y_predicted)
    RSS = np.sum(np.square(y_true - y_predicted))

    rse = math.sqrt(RSS / (len(y_true) - 2))
    
    return (y_true - y_predicted)/rse

def check_residuals(data: pd.DataFrame,
                    time_col: str,
                    y_true: str,
                    y_predicted: str,
                    selected: str,
                    data_group: str,
                    period = 'D',
                    diff = 0):

    temp = data[data[data_group] == selected].copy()
    
    if period == 'D':
        num_lags = (temp.shape[0]/2)-1
    elif period == 'M':
        num_lags = (temp.shape[0]/2)-1
    else:
        return None
    temp[time_col] = pd.to_datetime(temp[time_col], format = '%Y-%m-%d')
    
    temp.index = pd.DatetimeIndex(temp.index)
    
    temp = temp.resample(period).sum()

    temp['residual'] = temp[y_true] - temp[y_predicted]

    alpha = 0.01
        
    #stat, p = shapiro(temp['Residuo'])

    texto0 = f"{temp.shape[0]} dias"
    media = fr'média = {round(temp.residual.mean(),3)}'
    mediana = fr'mediana = {round(temp.residual.median(),3)}'
    desvio = fr'desvio padrão = {round(temp.residual.std(),3)}'
    #normal = "Normal:" + ("Sim" if (p > alpha) else "Não")

    fig = make_subplots(
    rows=2, cols=2, subplot_titles=("Autocorrelação", "Autocorrelação Parcial")
)

    # Add traces
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]), row=1, col=1)
    fig.add_trace(go.Scatter(x=[20, 30, 40], y=[50, 60, 70]), row=1, col=2)
    fig.add_trace(go.Scatter(x=[300, 400, 500], y=[600, 700, 800]), row=2, col=1)
    fig.add_trace(go.Scatter(x=[4000, 5000, 6000], y=[7000, 8000, 9000]), row=2, col=2)

    # Update xaxis properties
    fig.update_xaxes(title_text="xaxis 1 title", row=1, col=1)
    fig.update_xaxes(title_text="xaxis 2 title", range=[10, 50], row=1, col=2)
    fig.update_xaxes(title_text="xaxis 3 title", showgrid=False, row=2, col=1)
    fig.update_xaxes(title_text="xaxis 4 title", type="log", row=2, col=2)

    # Update yaxis properties
    fig.update_yaxes(title_text="yaxis 1 title", row=1, col=1)
    fig.update_yaxes(title_text="yaxis 2 title", range=[40, 80], row=1, col=2)
    fig.update_yaxes(title_text="yaxis 3 title", showgrid=False, row=2, col=1)
    fig.update_yaxes(title_text="yaxis 4 title", row=2, col=2)

    # Update title and height
    fig.update_layout(title_text="Customizing Subplot Axes", height=700)

    st.plotly_chart(fig, use_container_width=True)
        
def plot_series(data: pd.DataFrame,
                    time_col: str,
                    y_true: str,
                    y_predicted: str,
                    data_group: str,
                    selected: str,
                    # style: Dict[Any, Any],
                    period = 'D',
                    diff = 0
                    ) -> go.Figure:

    """Creates a plotly line plot showing forecasts and actual values on evaluation period.
    Parameters
    ----------
    eval_df : pd.DataFrame
        Dataframe com observado e previsto.
    style : Dict
        Style specifications for the graph (colors).
    Returns
    -------
    go.Figure
    Plotly line plot showing forecasts and actual values on evaluation period.
    """
    
    plot_df = data[data[data_group] == selected].copy()
    
    fig = px.line(plot_df,
                    x=time_col,
                    y=[y_true, y_predicted],
                    # color_discrete_sequence=style["colors"][1:],
                    hover_data={"variable": True, "value": ":.1f", time_col: False},
                )
    
    fig.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list(
                        [
                            dict(count=7, label="1w", step="day", stepmode="backward"),
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=3, label="3m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(step="all"),
                        ]
                    )
                ),
            )
    
    fig.update_layout(
                yaxis_title='Consumo',
                legend_title_text="",
                height=600,
                width=1000,
                title_text="Previsto vs Real",
                title_x=0.5,
                title_y=1,
                hovermode="x unified",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
        )
    
    st.plotly_chart(fig, use_container_width=True)
