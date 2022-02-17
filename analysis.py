from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import pacf, acf
import statsmodels.api as sm
from scipy.stats import shapiro
import math
import numpy as np
import pandas as pd
from pandas.plotting import autocorrelation_plot
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import List, Any, Dict, Tuple
from dashboard import * 

def preprocess_dataframe(data: pd.DataFrame,
                         data_group: str,
                         time_col: str,
                         y_true: str,
                         y_predicted: str,
                         ) -> pd.DataFrame:
    
    data[time_col] = pd.to_datetime(data[time_col],format = '%Y-%m-%d')
    data[time_col] = data[time_col].dt.date
    data['mape'] = MAPE(data[y_true],data[y_predicted])
    
    # Limiar do MAPE para evitar distorções
    data['mape'] = np.where(data['mape']>100, 100,data['mape'])
    data['mpe'] = MPE(data[y_true],data[y_predicted])
    data['residuo'] = data[y_true] - data[y_predicted]
    data['acima5'] = np.where(data['mape']>5, True, False)
    data['lim_sup'] = np.abs(data[y_true]*5/100)
    data['lim_inf'] = -1*np.abs(data[y_true]*5/100)
    data[y_true+'_diff'] = data[y_true].diff()
    data = data.sort_values(by = time_col, ascending=True)
    
    return data

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
        residual = y_true - y_predicted
        mape = np.where(y_true!=0, residual/y_true, np.nan)
        mape = np.where((residual==0) & (y_predicted==0), 0, mape)
        #mape = np.where(mape > 0.5, 0.1, mape)
        return 100*np.abs(mape)
    except:
        return 0   

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
    
def check_residuals(data: pd.DataFrame,
                    time_col: str,
                    selected: str,
                    data_group: str,
                    period = 'D'):

    data = data[data[data_group] == selected]
    #data.index = pd.DatetimeIndex(data.index)
    #data = data.resample(period).sum()
    
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=data[time_col],
                            y=data['residuo'],
                            mode='lines',
                            name='Resíduo'))
    fig.add_trace(go.Scattergl(
            y=data['lim_sup'], 
            x=data[time_col],
            line=dict(color='red'),
            opacity=0.45,
            name='+5%'))

    fig.add_trace(go.Scattergl(
            y=data['lim_inf'], 
            x=data[time_col],
            line=dict(color='red'),
            opacity=0.45,
            name='-5%'))
    
    fig.update_xaxes(title_text="Data")
    fig.update_yaxes(title_text= "Residuo", showgrid=False, zerolinecolor='#000000')
    fig = format_fig(fig, 'Série dos Resíduos', x_title=time_col, y_title='Resíduo')

    st.plotly_chart(fig, use_container_width=True)
    
    #fig = go.Figure()
    fig = px.histogram(data, x="residuo",
                   marginal="box", # or violin, rug
                   hover_data=['residuo'])
 
    #fig.add_trace(go.Histogram(x=data['residuo']))
    fig.update_traces(opacity=0.75)
    fig = format_fig(fig, 'Distribuição dos Resíduos', x_title='Resíduo', y_title='Contagem')
    st.plotly_chart(fig, use_container_width=True)
    
    corr_plot(data['residuo'])
    #corr_plot(data['residuo'], plot_pacf=True)
    
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
                height=PLOT_HEIGHT,
                width=PLOT_WIDTH,
                title_text="Previsto vs Real",
                title_x=0.5,
                title_y=1,
                hovermode="x unified"
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
def plot_daily_error(df, predictions, test, city_gate, limit=5):
    """Exibe gráfico com desvio percentual diário para cada predição realizada."""
    # Calculando PCS médio por city gate
    mean_pcs = df.groupby('city_gate').pcs.mean().to_dict()
    
    # Calculando QDR real e da predição
    min_test_date = test.index.min()
    qdr = df.query(f'city_gate == "{city_gate}"').set_index('data').qdr[min_test_date:]
    
    qdr_pred = {}
    for forecast_days in range(1, 4):
        qdr_pred[forecast_days] = (predictions[forecast_days] * mean_pcs[city_gate]/9400).dropna()
    
    error = pd.DataFrame()

    # Calculando desvios na predição dia a dia
    for days in range(1, 4):
        if error.empty:
            error = (((qdr_pred[days]/qdr) - 1) * 100).reset_index().rename(columns={'index': 'data', 0: days})
        else:
            error = error.merge((((qdr_pred[days]/qdr) - 1) * 100).reset_index().rename(columns={'index': 'data', 0: days}),
                                on='data', how='outer', validate='1:1')

    # Removendo datas sem ground truth
    error = error.set_index('data').dropna(how='all').reset_index()

    # Reestruturando DataFrame para gráfico
    error = error.melt(id_vars='data', var_name='dias', value_name='desvio')
    
    # Printando erro textualmente
    print('No total são', error.data.nunique(), 'dias de predição.')
    print()

    error['outside_range'] = error.desvio.abs() > limit
    outside_range = error.groupby('dias').outside_range.sum()

    print(f'Ficamos fora da margem de {limit}% de erro em:')
    for days in range(1, 4):
        print(f'{days} dias de antecedência: {outside_range[days]} dias ({int((outside_range[days]/error.data.nunique() * 100))}%)')
    print()
    
    # Gerando gráfico
    fig = px.line(error, x='data', y='desvio', color='dias', labels={'desvio': '% Desvio', 'data': '', 'dias': 'Dias'}, 
                  title='Desvio % do Previsto vs. Realizado - QDR')
    fig.add_hline(y=limit, line_color='gray', line_dash='dash')
    fig.add_hline(y=-limit, line_color='gray', line_dash='dash')
    
    return fig

def create_global_metrics(data, data_group):
    
    # Cálculo do MAPE
    data = data.groupby([data_group]).mean().reset_index()
    fig = go.Figure(data=[go.Bar(x=data[data_group].unique().tolist(),
                                 y=data["mape"], text = data["mape"])])
    # Customize aspect
    fig.update_traces(marker_color='rgb(32,4,114)', marker_line_color='rgb(157, 0, 25)',
                  marker_line_width=1.5, opacity=0.75, texttemplate='%{text:.3s}', textposition='outside')
    fig.update_layout(title_text='MAPE', hovermode='x')          
    #fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    #fig.update_xaxes(categoryorder='category ascending')
    fig = format_fig(fig, 'MAPE', x_title=data_group, y_title='Percentual')
    st.plotly_chart(fig, use_container_width=True)
    # Cálculo do Percentual Acima5%
    
def plot_error_distribution(test, predictions, city_gate, bin_limits, bin_size):
    """Exibe histograma com distribuição dos valores de erro."""
    err_df = pd.merge(test, predictions, left_index=True, right_index=True, how='inner')
    
    # Add histogram data
    x1 = err_df['volume'] - err_df[1]
    x2  = err_df['volume'] - err_df[2]
    x3  = err_df['volume'] - err_df[3]
    x1 = x1[pd.notna(err_df['volume'] - err_df[1])].values
    x2 = x2[pd.notna(err_df['volume'] - err_df[2])].values
    x3 = x3[pd.notna(err_df['volume'] - err_df[3])].values
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=x1, name='Erros de 1 dia a frente', xbins=dict(start=bin_limits[0], end=bin_limits[1], size=bin_size)
    ))
    
    fig.add_trace(go.Histogram(
        x=x2, name='Erros de 2 dias a frente', xbins=dict(start=bin_limits[0], end=bin_limits[1], size=bin_size)
    ))
    
    fig.add_trace(go.Histogram(
        x=x3, name='Erros de 3 dias a frente', xbins=dict(start=bin_limits[0], end=bin_limits[1], size=bin_size)
    ))
    
    # Overlay both histograms
    fig.update_layout(barmode='overlay', title=f'Distribuição dos erros na previsão - {city_gate}')
    
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.6)
    
    return fig

def corr_plot(series, plot_pacf=False):
    corr_array = pacf(series.dropna(), alpha=0.05) if plot_pacf else acf(series.dropna(), alpha=0.05, nlags=45)
    lower_y = corr_array[1][:,0] - corr_array[0]
    upper_y = corr_array[1][:,1] - corr_array[0]

    fig = go.Figure()
    [fig.add_scatter(x=(x,x), y=(0,corr_array[0][x]), mode='lines',line_color='#3f3f3f') 
     for x in range(len(corr_array[0]))]
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=corr_array[0], mode='markers', marker_color='#1f77b4',
                   marker_size=12)
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=upper_y, mode='lines', line_color='rgba(255,255,255,0)')
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=lower_y, mode='lines',fillcolor='rgba(32, 146, 230,0.3)',
            fill='tonexty', line_color='rgba(255,255,255,0)')
    fig.update_traces(showlegend=False)
    fig.update_xaxes(title_text="Lags", range=[-1, len(corr_array[0])])
    fig.update_yaxes(showgrid=False, zerolinecolor='#000000')
    
    title='Partial Autocorrelation (PACF)' if plot_pacf else 'Autocorrelation (ACF)'
    fig = format_fig(fig, title_text = title, x_title = 'Lags', y_title='Corr')
    #fig.update_layout(title=title)
    
    st.plotly_chart(fig, use_container_width=True)
    
def create_error_metrics(data: pd.DataFrame,
                time_col: str,
                selected: str,
                data_group: str,
                period = 'D'):

    data = data[data[data_group] == selected]
    #data.index = pd.DatetimeIndex(data.index)
    #data = data.resample(period).sum()
    
    num_lags = 45

    fig = make_subplots(
    rows=2, cols=2, subplot_titles=("1", "2", "3", "4")
    )

    plot_pacf = True
    corr_array = pacf(data['residuo'], alpha=0.05) if plot_pacf else acf(data['residuo'], alpha=0.05)
    lower_y = corr_array[1][:,0] - corr_array[0]
    upper_y = corr_array[1][:,1] - corr_array[0]

    [fig.add_scatter(x=(x,x), y=(0,corr_array[0][x]), mode='lines',line_color='#3f3f3f', row=1, col=2) 
    for x in range(len(corr_array[0]))]
    
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=corr_array[0], mode='markers', marker_color='#1f77b4',
                marker_size=12, row=1, col=2)
    
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=upper_y, mode='lines', line_color='rgba(255,255,255,0)', row=1, col=2)
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=lower_y, mode='lines',fillcolor='rgba(32, 146, 230,0.3)',
            fill='tonexty', line_color='rgba(255,255,255,0)', row=1, col=2)
    
    # Add traces
    
    #fig.add_trace(corr_plot(data['residuo']), row=1, col=2)
    fig.add_trace(go.Histogram(x=data['residuo']), row=2, col=1)
    #fig.add_trace(corr_plot(data['residuo'], plot_pacf = True), row=2, col=2)
    fig.add_trace(go.Scatter(x=data[time_col],
                            y=data['residuo'],
                            mode='lines',
                            name='Resíduo'), row=1, col=1)
    # Update xaxis properties
    fig.update_xaxes(title_text="Resíduo", row=1, col=1)
    fig.update_xaxes(title_text="ACF", range=[0, num_lags], row=1, col=2)
    fig.update_xaxes(title_text="Histograma", row=2, col=1)
    fig.update_xaxes(title_text="PACF", row=2, col=2)

    # Update yaxis properties
    fig.update_yaxes(title_text='Resíduo', showgrid=False, row=1, col=1)
    fig.update_yaxes(title_text="Lags", showgrid=False, row=1, col=2)
    fig.update_yaxes(title_text="Histograma", showgrid=False, row=2, col=1)
    fig.update_yaxes(title_text="Lags", showgrid=False, row=2, col=2)

    # Update title and height
    fig.update_layout(title_text="Avaliação dos Resíduos", height=700)

    st.plotly_chart(fig, use_container_width=True)
    