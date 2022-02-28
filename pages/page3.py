import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import pacf, acf
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.tseries.holiday import AbstractHolidayCalendar, GoodFriday, Holiday, Easter, Day
from typing import List, Any, Dict, Tuple

from utils import nomear_mes, nomear_dia

from kpi import *
from dashboard import *

def create_page3(df, data_group, time_col, y_true, y_predicted):
    pass

def standard_residual(data, data_group, y_true, y_predicted):
    for item in sorted(data[data_group].unique().tolist()):
        data.loc[data[data_group] == item, 'std_residuo'] = \
            RSE(data.loc[data[data_group] == item, y_true],
                data.loc[data[data_group] == item, y_predicted])
    return data

def check_residuals(data: pd.DataFrame,
                    time_col: str,
                    selected: str,
                    data_group: str,
                    period = 'D'):

    data = data[data[data_group] == selected]
    #data.index = pd.DatetimeIndex(data.index)
    #data = data.resample(period).sum()

    with st.expander("Série"):
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
                                name='Resíduo Padronizado'
                                )
            )
            fig.add_trace(go.Scatter(
                    y=data['lim_sup'], 
                    x=data[time_col],
                    line=dict(width=0),
                    mode='lines',
                    marker=dict(color="#444"),
                    showlegend=False,
                    name='lim sup'
                    )
            )
            fig.add_trace(go.Scatter(
                    y=data['lim_inf'], 
                    x=data[time_col],
                    mode='lines',
                    line=dict(width=0),
                    fillcolor='rgba(167, 0, 91, 0.2)',
                    fill='tonexty',
                    showlegend=False,
                    marker=dict(color="#444"),
                    name='lim inf'
                    )
            )
        else:
            fig.add_trace(go.Scattergl(x=data[time_col],
                                    y=data['residuo'],
                                    mode='lines',
                                    line=dict(color='rgb(32,4,114)'),
                                    name='Resíduo'
                                    )
            )
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
                                    name='MPE'
                                    )
        )
        fig.add_trace(go.Scattergl(
                    y=data['lim_sup'], 
                    x=data[time_col],
                    line=dict(color='red', dash = 'dash'),
                    name='+5%'
                    )
        )
        fig.add_trace(go.Scattergl(
                    y=data['lim_inf'], 
                    x=data[time_col],
                    line=dict(color='red', dash = 'dash'),
                    name='-5%'
                    )
        )
        fig.update_xaxes(title_text="Data")
        fig.update_yaxes(title_text= "Erro Médio Percentual", showgrid=False, zerolinecolor='#000000')
        fig = format_fig(fig, '', x_title=time_col, y_title='Erro Médio Percentual')
        st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("Medidas de Posição"):

        fig = ff.create_distplot([data['residuo']], ['residuo'],
                                 show_hist=False, 
                                 colors=['rgb(32,4,114)']
                                )
        
        fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
        check_seasonal_residuals(data, time_col, selected, data_group)

    with st.expander("Função de Autocorrelação"):
        corr_lin = data['residuo'].dropna()
        corr_quad = corr_lin**2
        p_acf = st.checkbox('Autocorrelação Parcial')
        st.write('Resíduos')
        corr_plot(corr_lin, plot_pacf = p_acf)
        st.write('Resíduos Quadráticos')
        corr_plot(corr_quad, plot_pacf = p_acf)
        
def check_seasonal_residuals(data: pd.DataFrame,
                    time_col: str,
                    selected: str,
                    data_group: str
                    ):
    
    # Monthly Boxplot
    st.write('Resíduos por Mês')
    df_month = data[data[data_group] == selected]
    df_month[time_col] = pd.to_datetime(df_month[time_col], format='%Y-%m-%d')
    df_month['month'] = df_month[time_col].dt.month
    df_month['month'] = df_month['month'].apply(nomear_mes)
    
    fig = go.Figure()
    fig.add_trace(go.Box(
        x = df_month["month"],
        y = df_month["residuo"],
        boxmean=True
        )
    )
    fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
    st.plotly_chart(fig, use_container_width=True)
    
    # Weekday Boxplot
    st.write('Resíduos por Dia da Semana')
    df_weekday = data[data[data_group] == selected]
    df_weekday[time_col] = pd.to_datetime(df_weekday[time_col], format='%Y-%m-%d')
    df_weekday['weekday'] = df_weekday[time_col].dt.weekday
    df_weekday.sort_values(by = 'weekday', inplace = True)
    df_weekday['weekday'] = df_weekday['weekday'].apply(nomear_dia)
    
    fig = go.Figure()
    fig.add_trace(go.Box(
        x = df_weekday["weekday"],
        y = df_weekday["residuo"],
        #name='Only Mean',
        boxmean=True #    boxmean='sd' # represent mean and standard deviation
        )
    )
    fig.update_traces(quartilemethod="exclusive")
    st.plotly_chart(fig, use_container_width=True)

def check_mape(data: pd.DataFrame,
                    time_col: str,
                    selected: str,
                    data_group: str
                    ):
    
    st.subheader(f'MAPE do {data_group}')
    data = generate_holidays(data, time_col, selected,data_group)
    data['isholiday'] = np.where(data['feriado'].isna(), 0 , 1)

    dfplot = data.loc[data[data_group] == selected, [data_group,time_col,'mape','mpe']].sort_values(by = ['mape'], ascending=False)
    dfplot['dia_da_semana'] = pd.to_datetime(dfplot[time_col], format='%Y-%m-%d').dt.weekday.apply(nomear_dia)
    #dfplot = dfplot.dropna()
    
    with st.expander("Medidas de Posição"):
        st.dataframe(dfplot[[data_group, time_col, 'mape']])
        st.write('Distribuição MPE')
        fig = ff.create_distplot([dfplot['mpe']], ['mpe'],
                                    show_hist=False, 
                                    colors=['rgb(32,4,114)'])
        fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly Boxplot
        st.write('MAPE - Mês')
        df_month = data[data[data_group] == selected]
        df_month[time_col] = pd.to_datetime(df_month[time_col], format='%Y-%m-%d')
        df_month['month'] = df_month[time_col].dt.month
        df_month['month'] = df_month['month'].apply(nomear_mes)
        
        fig = go.Figure()
        fig.add_trace(go.Box(
            x = df_month["month"],
            y = df_month["mape"],
            boxmean=True
            )
        )
        fig.update_xaxes(tickangle=-45)
        fig.update_traces(quartilemethod="exclusive") 
        st.plotly_chart(fig, use_container_width=True)
        
        # Weekday Boxplot
        st.write('MAPE - Dias da Semana')
        df_weekday = data[data[data_group] == selected]
        df_weekday[time_col] = pd.to_datetime(df_weekday[time_col], format='%Y-%m-%d')
        df_weekday['weekday'] = df_weekday[time_col].dt.weekday
        df_weekday.sort_values(by = 'weekday', inplace = True)
        df_weekday['weekday'] = df_weekday['weekday'].apply(nomear_dia)
        
        fig = go.Figure()
        fig.add_trace(go.Box(
            x = df_weekday["weekday"],
            y = df_weekday["mape"],
            boxmean=True
            )
        )
        fig.update_traces(quartilemethod="exclusive")
        st.plotly_chart(fig, use_container_width=True)     

    with st.expander(f"Feriados (todos os {data_group})"):
        st.dataframe(data[[data_group, time_col, 'mape', 'feriado']])
        dfplot = data[data['isholiday'] == 1]
        data = data[data[data_group] == selected]
        
        fig = go.Figure()
        fig.add_trace(go.Box(
            x = dfplot["feriado"],
            y = dfplot["mape"],
            boxmean=True
            )
        )
        fig.update_xaxes(tickangle=-45)
        fig.update_traces(quartilemethod="exclusive")
        st.plotly_chart(fig, use_container_width=True)  
    
def check_rmse(data: pd.DataFrame,
                    time_col: str,
                    selected: str,
                    data_group: str
                    ):
    
    st.subheader('RMSE')
    dfplot = data.loc[data[data_group] == selected, [data_group,time_col,'rmse']].sort_values(by = ['rmse'], ascending=False)
    dfplot['dia_da_semana'] = pd.to_datetime(dfplot[time_col], format='%Y-%m-%d').dt.weekday.apply(nomear_dia)
    dfplot = dfplot.dropna()
    #st.dataframe(dfplot)
    
    with st.expander("Medidas de Posição"):
        # Monthly Boxplot
        st.write('RMSE - Mês')
        df_month = data[data[data_group] == selected]
        df_month[time_col] = pd.to_datetime(df_month[time_col], format='%Y-%m-%d')
        df_month['month'] = df_month[time_col].dt.month
        df_month['month'] = df_month['month'].apply(nomear_mes)
        fig = go.Figure()
        fig.add_trace(go.Box(
            x = df_month["month"],
            y = df_month["rmse"],
            boxmean=True
            )
        )
        fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
        st.plotly_chart(fig, use_container_width=True)
        
        # Weekday Boxplot
        st.write('RMSE - Dias da Semana')
        df_weekday = data[data[data_group] == selected]
        df_weekday[time_col] = pd.to_datetime(df_weekday[time_col], format='%Y-%m-%d')
        df_weekday['weekday'] = df_weekday[time_col].dt.weekday
        df_weekday.sort_values(by = 'weekday', inplace = True)
        df_weekday['weekday'] = df_weekday['weekday'].apply(nomear_dia)
        fig = go.Figure()
        fig.add_trace(go.Box(
            x = df_weekday["weekday"],
            y = df_weekday["rmse"],
            boxmean=True
            )
        )
        fig.update_traces(quartilemethod="exclusive")
        st.plotly_chart(fig, use_container_width=True)  
        
def generate_holidays(data: pd.DataFrame,
                   time_col:str,
                   selected:str,
                   data_group: str
                    ):

    class Feriados_SP(AbstractHolidayCalendar):
        rules = [
            Holiday('Confraternização Universal', month=1, day=1),
            Holiday('Aniversário de São Paulo', month=1, day=25),
            Holiday('Segunda-Feira de Carnaval', month=1, day=1, offset=[Easter(), Day(-48)]),
            Holiday('Terça-Feira de Carnaval', month=1, day=1, offset=[Easter(), Day(-47)]),
            Holiday('Quarta-Feira de Cinzas', month=1, day=1, offset=[Easter(), Day(-46)]),
            # Sexta-feira Santa
            GoodFriday,
            Holiday('Corpus Christi', month=1, day=1, offset=[Easter(), Day(60)]),
            Holiday('Tiradentes', month = 4, day = 21),
            Holiday('Dia do Trabalho', month = 5, day = 1),
            Holiday('Revolução Constitucionalista', month=7, day=9, start_date='1997-01-01'),
            Holiday('Independência do Brasil', month = 9, day = 7),
            Holiday('Nossa Senhora Aparecida', month = 10, day = 12),
            Holiday('Finados', month = 11, day = 2),
            Holiday('Proclamação da República', month = 11, day = 15),
            Holiday('Dia da Consciencia Negra', month=11, day=20, start_date='2004-01-01'),
            Holiday('Vespera de Natal', month=12, day=24),
            Holiday('Natal', month = 12, day = 25)
            ]

    dferiado = data
    #dferiado = data[data[data_group]==selected]
    
    dferiado[time_col] = pd.to_datetime(dferiado[time_col], format = '%Y-%m-%d')
    sp_cal = Feriados_SP()
    feriados_sp = sp_cal.holidays(dferiado[time_col].min(), dferiado[time_col].max(), return_name = True)
    feriados_sp = feriados_sp.reset_index()
    feriados_sp.columns = [time_col, 'feriado']
    dferiado = dferiado.merge(feriados_sp, on = [time_col], how = 'left')
    
    return dferiado

def plot_seasonal_decompose(df, data_group, selected, time_col, col, decompose_model = 'additive', interpol_method='linear', shared_y=False):
    """Realiza decomposição automática da série temporal e imprime os quatro gráficos resultantes
    (série, tendência, sazonalidade e resíduos)."""

    assert decompose_model in ['additive', 'multiplicative']
    
    # Reindexando série para frequência diária
    cg_vol = df.loc[df[data_group] == selected].set_index(time_col).asfreq('D')[col]
    
    # Realizando interpolação linear de dias faltantes
    cg_vol = cg_vol.interpolate(method=interpol_method)
    
    # Decomposição linear
    result = seasonal_decompose(cg_vol, model=decompose_model)
    decompose_model = 'Aditiva' if decompose_model == 'additive' else 'Multiplicativa'
    
    # Plotando decomposição aditiva
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, shared_yaxes=shared_y, vertical_spacing=0.05, 
                        subplot_titles=('Série', 'Tendência', 'Sazonalidade', 'Residual'))
    fig.update_layout(title=f'Decomposição {decompose_model} - {selected}', showlegend=False, height=800, width=1200)

    fig.add_scatter(row=1, col=1, y=cg_vol, x=cg_vol.index, name='Série')
    fig.add_scatter(row=2, col=1, y=result.trend, x=result.trend.index, name='Tendência')
    fig.add_scatter(row=3, col=1, y=result.seasonal, x=result.trend.index, name='Sazonalidade')
    fig.add_scatter(row=4, col=1, y=result.resid, x=result.trend.index, name='Residual')
    
    st.plotly_chart(fig, use_container_width=True)

def corr_plot(series, plot_pacf=False):
    # Sem a Autocorrelation do Lag 0
    corr_array = pacf(series.dropna(), alpha=0.05, nlags=45) if plot_pacf else acf(series.dropna(), alpha=0.05, nlags=45)
    lower_y = corr_array[1][:,0] - corr_array[0]
    upper_y = corr_array[1][:,1] - corr_array[0]
    
    fig = go.Figure()
    [fig.add_scatter(x=(x,x), y=(0,corr_array[0][x]), mode='lines',line_color='#3f3f3f',hoverinfo='skip') 
     for x in range(1, len(corr_array[0]))]
    fig.add_scatter(x=np.arange(1,len(corr_array[0])), y=corr_array[0][1:], mode='markers', marker_color='#1f77b4',
                   marker_size=12)
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=upper_y[1:], mode='lines', line_color='rgba(255,255,255,0)', hoverinfo='skip')
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=lower_y[1:], mode='lines',fillcolor='rgba(32, 146, 230,0.3)', hoverinfo='skip',
            fill='tonexty', line_color='rgba(255,255,255,0)')
    fig.update_traces(showlegend=False)
    fig.update_xaxes(title_text="Lags", range=[-1, len(corr_array[0])])
    fig.update_yaxes(showgrid=False, zerolinecolor='#000000')

    title='Partial Autocorrelation (PACF)' if plot_pacf else 'Autocorrelation (ACF)'
    fig = format_fig(fig, title_text = title, x_title = 'Lags', y_title='Corr')
    
    st.plotly_chart(fig, use_container_width=True)

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
    
    dfplot = data[data[data_group] == selected].copy()
    fig = go.Figure(data=go.Scatter(x=dfplot[time_col],
                                    y=dfplot[y_true],
                                    mode='lines',
                                    name='Real',
                                    line_color='rgb(32,4,114)',
                                    )
                    )
    fig.add_trace(go.Scatter(x=dfplot[time_col],
                             y=dfplot[y_predicted],
                             mode='lines',
                             name='Previsto',
                             line_color='rgb(234, 82, 111)',
                             )
                  )
    fig.add_trace(go.Scatter(x=dfplot[time_col],
                            y=dfplot[y_true]-dfplot[y_predicted],
                            mode='lines',
                            name='Resíduo',
                            line_color='rgb(37, 206, 209)',
                            )
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
