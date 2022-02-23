#from scipy.stats import shapiro
#from pandas.plotting import autocorrelation_plot
#from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import pacf, acf
import statsmodels.api as sm
import streamlit as st
from typing import List, Any, Dict, Tuple
from kpi import MAPE, RMSE, MPE, RSE

from dashboard import * 

MESES =["Janeiro", "Fevereiro", "Março", "Abril",
          "Maio", "Junho", "Julho", "Agosto",
          "Setembro", "Outubro", "Novembro", "Dezembro"]

WEEKDAY = ["Segunda", "Terça",
           "Quarta","Quinta",
           "Sexta", "Sábado", "Domingo"]

nomear_dia = lambda x: WEEKDAY[x]
nomear_mes = lambda x: MESES[x-1]

def load_csv_file():
    
    with st.sidebar.expander("Leitura de arquivo"):    
        data_file = st.file_uploader("Selecionar arquivo CSV",type=["csv"])
        if data_file is not None:
            file_details = {"nome do arquivo":data_file.name,
                    "tipo do arquivo":data_file.type,
                    "tamanho do arquivo":data_file.size}

            df = pd.read_csv(data_file, parse_dates=True)
            #with st.expander("Informações dos dados:"):
            #    st.write(file_details)
        
            data_group = st.selectbox("Chave ID:", df.columns)
            time_col = st.selectbox("Coluna Temporal:", df.columns)
            y_true = st.selectbox("Série Real:", df.columns)
            y_predicted = st.selectbox("Série Prevista:", df.columns)
            data_group2 = st.selectbox("Agrupamento:", df.columns)
            chosen_group = st.selectbox(f"Selecione o agrupamento:",
                                    sorted(df[data_group2].unique().tolist()))

            try:
                df = df[df[data_group2]==chosen_group]
                df = preprocess_dataframe(df,
                                    time_col,
                                    y_true,
                                    y_predicted)

                return df, time_col, data_group, y_true, y_predicted
            
            except:
                pass
            
def preprocess_dataframe(data: pd.DataFrame,
                         time_col: str,
                         y_true: str,
                         y_predicted: str,
                         ) -> pd.DataFrame:
    
    # Timestamp errado
    data[time_col] = pd.to_datetime(data[time_col], format = '%Y-%m-%d')
    data[time_col] = data[time_col].dt.date
    data = data.dropna()
    data['mape'] = MAPE(data[y_true],data[y_predicted])
    #data['rmse'] = RMSE(data[y_true],data[y_predicted])
    # Limiar do MAPE para evitar distorções
    data['mape'] = np.where(data['mape']>100, 100, data['mape'])
    data['mpe'] = MPE(data[y_true],data[y_predicted])
    data['residuo'] = data[y_true] - data[y_predicted]
    data['acima5'] = np.where(data['mape']>5, True, False)
    #data['lim_sup'] = 1.96
    #data['lim_inf'] = -1.96
    data[y_true+'_diff'] = data[y_true].diff()
    data = data.sort_values(by = time_col, ascending=True)
    return data

def standard_residual(data, data_group, y_true, y_predicted):
    for item in sorted(data[data_group].unique().tolist()):
        data.loc[data[data_group] == item, 'std_residuo'] = \
            RSE(data.loc[data[data_group] == item, y_true],
                data.loc[data[data_group] == item, y_predicted])
    return data

def create_global_metrics(data, data_group, y_true:str, y_predicted:str):
    
    #categories = data.groupby([data_group]).unique().tolist()
    #st.markdown(f"""
    #        <marquee style='width: 100%; color: rgb(234, 82, 111);'><b> <font size="5">{categories}.</b></font></marquee>""",
    #unsafe_allow_html = True)
    
    #DIAS ACIMA DE 5%
    st.markdown("""
            <span style="color:rgb(234, 82, 111)"><font size="5">DIAS ACIMA DE 5%</font></span>""",
    unsafe_allow_html = True)
    dfplot = data.groupby([data_group]).apply(lambda x: 100*x.acima5.sum()/x.acima5.count()).reset_index()
    fig = go.Figure(data=[go.Bar(x=dfplot[data_group].unique().tolist(),
                                y=dfplot.iloc[:, 1], text = dfplot.iloc[:,[1]])])
    # Customize aspect
    fig.update_xaxes(tickangle=-45)
    fig.update_traces(marker_color='rgb(234, 82, 111)', marker_line_color='rgb(0, 0, 0)',
                marker_line_width=1.5, opacity=0.75, texttemplate='%{text:.1f}', textposition='outside')
    fig.update_layout(hovermode='x')          
    #fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    #fig.update_xaxes(categoryorder='category ascending')
    fig = format_fig(fig, x_title=data_group, y_title='Percentual(%)')
    st.plotly_chart(fig, use_container_width=True)
    
    #MAPE
    st.markdown("""
            <span style="color:rgb(110, 68, 255)"><font size="5">MAPE</font></span>""",
    unsafe_allow_html = True)
    dfplot = data.groupby([data_group]).mean().reset_index()
    fig = go.Figure(data=[go.Bar(x=dfplot[data_group].unique().tolist(),
                                y=dfplot["mape"])])
    # Customize aspect
    fig.update_xaxes(tickangle=-45)
    fig.update_traces(marker_color='rgb(110, 68, 255)', marker_line_color='rgb(0, 0, 0)',
                marker_line_width=1.5, opacity=0.75,
                texttemplate='%{y:.1f}', textposition='outside')
    fig.update_layout(hovermode='x')          
    fig = format_fig(fig, x_title=data_group, y_title='Percentual(%)')
    st.plotly_chart(fig, use_container_width=True)

    #RMSE
    st.markdown("""
            <span style="color:rgb(37, 206, 209)"><font size="5">RMSE</font></span>""",
    unsafe_allow_html = True)
    dfplot = data.groupby([data_group]).apply(lambda x: RMSE(x[y_true], x[y_predicted])).reset_index()
    fig = go.Figure(data=[go.Bar(x=dfplot[data_group].unique().tolist(),
                                y=dfplot.iloc[:, 1], text = dfplot.iloc[:,[1]])])
    # Customize aspect
    fig.update_xaxes(tickangle=-45)
    fig.update_traces(marker_color='rgb(37, 206, 209)', marker_line_color='rgb(0, 0, 0)',
                marker_line_width=1.5, opacity=0.75, texttemplate='%{text:.0f}', textposition='outside')
    fig.update_layout(hovermode='x')          
    #fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    #fig.update_xaxes(categoryorder='category ascending')
    fig = format_fig(fig, x_title=data_group, y_title='rmse')
    st.plotly_chart(fig, use_container_width=True)

def create_grouped_radar(data, data_group, data_group2, time_col, y_true:str, y_predicted:str):
    categories = sorted(data[data_group].unique().tolist())
    groups = sorted(data[data_group2].unique().tolist())
    
    options = st.multiselect(
     'Adicione os Itens:',
    categories)

    chosen_metric=st.selectbox('Métrica', ['MAPE', 'ACIMA5'])
    #st.markdown("""
    #        <span style="color:rgb(32,4,114)"> <font size="5">MAPE</font></span>""",
    #unsafe_allow_html = True)
    fig = go.Figure()
    for group in groups:
        
        dfplot = data[data[data_group2]==group]
        dfplot = preprocess_dataframe(dfplot, time_col, y_true, y_predicted)
        
        if chosen_metric=='MAPE':
            values = dfplot.groupby([data_group]).mape.mean()
        else:
            values = dfplot.groupby([data_group]).apply(lambda x: 100*x.acima5.sum()/x.acima5.count())
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=sorted(options),
            fill='toself',
            opacity=0.65,
            mode = 'lines+markers',
            name=group
        ))
    fig.update_layout(
        template="plotly_white",
        polar=dict(
            radialaxis=dict(
            visible=True,
        )),
    showlegend=True,
    height=750, width=750,
    )
    st.plotly_chart(fig, use_container_width=True)

def check_residuals(data: pd.DataFrame,
                    time_col: str,
                    selected: str,
                    data_group: str,
                    period = 'D'):

    data = data[data[data_group] == selected]
    #data.index = pd.DatetimeIndex(data.index)
    #data = data.resample(period).sum()
    
    with st.expander("Resíduos"):
        standardize = st.checkbox('Resíduo Padronizado')
        fig = go.Figure()
        if standardize:
            data['lim_sup'] = 1.96
            data['lim_inf'] = -1*data['lim_sup']
            fig.add_trace(go.Scatter(x=data[time_col],
                                y=data['std_residuo'],
                                mode='lines',
                                line=dict(color='rgb(32,4,114)'),
                                showlegend=False,
                                name='Resíduo Padronizado'))
            fig.add_trace(go.Scatter(
                    y=data['lim_sup'], 
                    x=data[time_col],
                    line=dict(width=0),
                    mode='lines',
                    marker=dict(color="#444"),
                    showlegend=False,
                    name='lim sup'))
            fig.add_trace(go.Scatter(
                    y=data['lim_inf'], 
                    x=data[time_col],
                    mode='lines',
                    line=dict(width=0),
                    fillcolor='rgba(167, 0, 91, 0.2)',
                    fill='tonexty',
                    showlegend=False,
                    marker=dict(color="#444"),
                    name='lim inf'))
        else:
            fig.add_trace(go.Scattergl(x=data[time_col],
                                    y=data['residuo'],
                                    mode='lines',
                                    line=dict(color='rgb(32,4,114)'),
                                    name='Resíduo'))
            
        fig.update_xaxes(title_text="Data")
        fig.update_yaxes(title_text= "Residuo", showgrid=False, zerolinecolor='#000000')
        fig = format_fig(fig, '', x_title=time_col, y_title='Resíduo')
        st.plotly_chart(fig, use_container_width=True)

        fig = go.Figure()
        data['lim_sup'] = 5
        data['lim_inf'] = -1*data['lim_sup']

        fig.add_trace(go.Scattergl(x=data[time_col],
                                    y=data['mpe'],
                                    mode='markers',
                                    line=dict(color='rgb(32,4,114)'),
                                    name='MPE'))
        
        fig.add_trace(go.Scattergl(
                    y=data['lim_sup'], 
                    x=data[time_col],
                    line=dict(color='red', dash = 'dash'),
                    name='+5%'))

        fig.add_trace(go.Scattergl(
                    y=data['lim_inf'], 
                    x=data[time_col],
                    line=dict(color='red', dash = 'dash'),
                    name='-5%'))

        fig.update_xaxes(title_text="Data")
        fig.update_yaxes(title_text= "Erro Médio Percentual", showgrid=False, zerolinecolor='#000000')
        fig = format_fig(fig, '', x_title=time_col, y_title='Erro Médio Percentual')
        st.plotly_chart(fig, use_container_width=True)
        
    with st.expander("Distribuição"):
        #fig = px.histogram(data, x="residuo",
        #            marginal="box", # or violin, rug
        #            hover_data=['residuo'])
    
        #fig.update_traces(opacity=0.75)
        #fig = format_fig(fig, 'Distribuição dos Resíduos', x_title='Resíduo', y_title='Contagem')
        #st.plotly_chart(fig, use_container_width=True)

        fig = ff.create_distplot([data['residuo']], ['residuo'],
                                 show_hist=False, 
                                 colors=['rgb(32,4,114)'])
        fig.add_vline(x=2.5, line_width=1, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
        
    with st.expander("Função de Autocorrelação"):
        p_acf = st.checkbox('Autocorrelação Parcial')
        corr_plot(data['residuo'], plot_pacf = p_acf)

def check_seasonal_residuals(data: pd.DataFrame,
                    time_col: str,
                    selected: str,
                    data_group: str
                    ):
    
    # Monthly Boxplot
    with st.expander("Mês"):
        df_month = data[data[data_group] == selected]
        df_month[time_col] = pd.to_datetime(df_month[time_col], format='%Y-%m-%d')
        df_month['month'] = df_month[time_col].dt.month
        df_month['month'] = df_month['month'].apply(nomear_mes)
        
        fig = px.box(df_month, x="month", y="residuo")
        fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
        st.plotly_chart(fig, use_container_width=True)
    
    # Weekday Boxplot
    with st.expander("Dia da Semana"):
        df_weekday = data[data[data_group] == selected]
        df_weekday[time_col] = pd.to_datetime(df_weekday[time_col], format='%Y-%m-%d')
        df_weekday['weekday'] = df_weekday[time_col].dt.weekday
        df_weekday.sort_values(by = 'weekday', inplace = True)
        df_weekday['weekday'] = df_weekday['weekday'].apply(nomear_dia)
        fig = px.box(df_weekday, x="weekday", y="residuo")
        fig.update_traces(quartilemethod="exclusive")
        st.plotly_chart(fig, use_container_width=True)
    
def check_holidays(data: pd.DataFrame,
                    time_col: str,
                    selected: str,
                    data_group: str
                    ):
    
    # hOLIDAYS
    pass

def plot_series(data: pd.DataFrame,
                    time_col: str,
                    y_true: str,
                    y_predicted: str,
                    data_group: str,
                    selected: str
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
    fig = go.Figure(data=go.Scatter(x=plot_df[time_col],
                                    y=plot_df[y_true],
                                    mode='lines',
                                    name='Real',
                                    line_color='rgb(32,4,114)')
                    )
    fig.add_trace(go.Scatter(x=plot_df[time_col],
                             y=plot_df[y_predicted],
                             mode='lines',
                             name='Previsto',
                             line_color='rgb(234, 82, 111)')
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

def corr_plot(series, plot_pacf=False):
    corr_array = pacf(series.dropna(), alpha=0.05, nlags=45) if plot_pacf else acf(series.dropna(), alpha=0.05, nlags=45)
    lower_y = corr_array[1][:,0] - corr_array[0]
    upper_y = corr_array[1][:,1] - corr_array[0]

    fig = go.Figure()
    [fig.add_scatter(x=(x,x), y=(0,corr_array[0][x]), mode='lines',line_color='#3f3f3f',hoverinfo='skip') 
     for x in range(len(corr_array[0]))]
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=corr_array[0], mode='markers', marker_color='#1f77b4',
                   marker_size=12)
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=upper_y, mode='lines', line_color='rgba(255,255,255,0)', hoverinfo='skip')
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=lower_y, mode='lines',fillcolor='rgba(32, 146, 230,0.3)', hoverinfo='skip',
            fill='tonexty', line_color='rgba(255,255,255,0)')
    fig.update_traces(showlegend=False)
    fig.update_xaxes(title_text="Lags", range=[-1, len(corr_array[0])])
    fig.update_yaxes(showgrid=False, zerolinecolor='#000000')

    title='Partial Autocorrelation (PACF)' if plot_pacf else 'Autocorrelation (ACF)'
    fig = format_fig(fig, title_text = title, x_title = 'Lags', y_title='Corr')
    
    st.plotly_chart(fig, use_container_width=True)
