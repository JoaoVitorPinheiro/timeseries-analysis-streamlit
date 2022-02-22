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