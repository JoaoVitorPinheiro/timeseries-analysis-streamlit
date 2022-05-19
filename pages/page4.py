import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Any, Dict, Tuple

from kpi import *
from dashboard import *

def create_benchmark_view(df:pd.DataFrame,
                          time_col:str,
                          data_group:str,
                          classe:str,
                          y_true:str,
                          y_benchmark:str):
          
    all_items = sorted(df[data_group].unique().tolist())
    
    restricted_words = ['Comgás']
    y_bench = [i for i in y_benchmark if i not in restricted_words]
    custom = st.checkbox('Adicionar Híbrido:')
    #st.write(y_bench)
    
    df_mix = df.copy()
    #df_mix = df[df[data_group].isin(group_items)].copy()
    df_mix['mix'] = df_mix[y_bench[0]]
    
    model_names = ['LGBM_PCS', 'Extrap_PCS']
    
    if custom:
        # ALTERAR ESSA ESTRUTURA DE LISTA
        mixsetup = {
                'CG01': model_names[0],
                'CG02': model_names[1],
                'CG03': model_names[1],
                'CG04': model_names[0],
                'CG05': model_names[1],
                'CG06': model_names[0],
                'CG07': model_names[1],
                'CG08': model_names[0],
                'CG09': model_names[1],
                'CG10': model_names[1],
                'CG11': model_names[1],
                'CG12': model_names[0],
                'CG13': model_names[1],
                'CG14': model_names[1],
                'CG15': model_names[1],
                'CG16': model_names[0],
                'CG17': model_names[0],
                'CG18': model_names[1],
                'CG19': model_names[0],
                'CG20': model_names[1],
                'CG21': model_names[1],
                'CG22': model_names[1],
                'CG23': model_names[0],
                'CG24': model_names[0]
                }
        
        with st.expander(f'Modelo por {data_group}:'):

            for col in all_items:
                mixsetup[col] = st.selectbox(f'{col}:', [mixsetup[col]] + list(filter(lambda x: x != mixsetup[col], y_bench)))
                df_mix.loc[df_mix[data_group] == col, 'mix'] = df_mix.loc[df_mix[data_group] == col, mixsetup[col]]
            
            #st.dataframe(df_mix.loc[df_mix[data_group].isin(group_items), [time_col, data_group, classe, 'mix']+y_benchmark])
        df['Híbrido'] = df_mix['mix']
        y_benchmark.append('Híbrido')
        st.write(mixsetup)
    
    st.session_state['chosen_item'] = st.selectbox(f'{classe}', sorted(df[classe].unique().tolist()))
    group_items = sorted(df.loc[df[classe] == st.session_state['chosen_item'], data_group].unique().tolist())
    
    st.write({f'{data_group}s da Zona de Entrega': group_items})
    benchmark_df = df.copy()
    benchmark_df[time_col] = pd.to_datetime(benchmark_df[time_col])
    benchmark_df = benchmark_df.groupby([pd.Grouper(key = time_col, freq = 'D'), classe]).sum().reset_index()
    benchmark_df = benchmark_df.reset_index()
    days_count = benchmark_df.loc[benchmark_df[classe] == st.session_state['chosen_item']].shape[0] 
    
    fig_series = go.Figure()
    fig_scatter = go.Figure()
    #fig.update_xaxes(title_text="Data")
    #fig.update_yaxes(title_text= "Erro Médio Percentual", showgrid=False, zerolinecolor='#000000')
    
    benchmark_df['lim_sup'] = 5
    benchmark_df['lim_inf'] = -1*benchmark_df['lim_sup']
    
    rgb_list = [
                'rgb(216, 71, 151)',
                'rgb(6, 214, 160)',
                'rgb(107, 212, 37)',
                'rgb(81, 88, 187)',
                'rgb(195, 31, 31)',
                'rgb(65, 151, 151)',
                'rgb(222, 158, 54)']  
    
    for num, prev in enumerate(y_benchmark):
    
        benchmark_df['residuo'] = benchmark_df[prev] - benchmark_df[y_true]
        benchmark_df['mpe'] = 100*(benchmark_df['residuo']/benchmark_df[prev])
        benchmark_df['mpe'] = benchmark_df['mpe'].clip(-100,100)
        benchmark_df['mape'] = np.abs(benchmark_df['mpe'])
        benchmark_df['acima5'] = np.where(benchmark_df['mape']>5, 1, 0)
        benchmark_df['acima20'] = np.where(benchmark_df['mape']>20, 1, 0)
        
        dfplot = benchmark_df.loc[benchmark_df[classe] == st.session_state['chosen_item']]
        
        days_count = dfplot.shape[0]
        mape_metrica = dfplot.mape.clip(0,100).mean()
        
        acima5_mask = (dfplot['acima5']==True)
        days_acima5 = dfplot.loc[acima5_mask].shape[0]
        perc_acima5 = days_acima5/days_count
        
        acima20_mask = (dfplot['acima20']==True)
        days_acima20 = dfplot.loc[acima20_mask].shape[0] 
        perc_acima20 = days_acima20/days_count
        
        col2 = st.columns(4)
        delta1 = np.round(mape_metrica-5,2)
        
        col2[0].metric(label="Previsto", value=f"{prev}")
        
        col2[1].metric(label="MAPE",
                    value=f"{round(mape_metrica,2)}%",
                    delta=f"{delta1}%",
                    delta_color="inverse")
        
        col2[2].metric(label="Dias Acima de 5%",
                    value=f"{round(100*perc_acima5,2)}%",
                    delta=f"{days_acima5} dias",
                    delta_color='off')
        
        col2[3].metric(label="Dias Acima de 20%",
                    value=f"{round(100*perc_acima20,2)}%",
                    delta=f"{days_acima20} dias",
                    delta_color='off')
        # VOLUME
        fig_series.add_trace(go.Scattergl(x= dfplot[time_col],
                                    y= dfplot[prev],
                                    mode='lines+markers',
                                    line=dict(color=rgb_list[num]),
                                    name= prev
                                    )
        )
        # ERRO
        fig_scatter.add_trace(go.Scattergl(x=dfplot[time_col],
                                y=dfplot['mpe'],
                                mode='markers',
                                line=dict(color=rgb_list[num]),
                                name= prev
                                )
        )
    
    with st.expander(f'Visualização do volume previsto e erros: {classe}'):
        fig_series.add_trace(go.Scattergl(x= dfplot[time_col],
                                        y= dfplot[y_true],
                                        mode='lines',
                                        line=dict(color='rgb(0, 0, 0)', dash = 'dash'),
                                        name='Real'
                                        )
        )
        fig_series.update_xaxes(title_text="Data")
        fig_series.update_yaxes(title_text= "Residuo", showgrid=False, zerolinecolor='#000000')
        fig_series = format_fig(fig_series, '', x_title=time_col, y_title='Volume')
        st.plotly_chart(fig_series, use_container_width=True)
            
        fig_scatter.add_trace(go.Scattergl(
                    y=dfplot['lim_sup'], 
                    x=dfplot[time_col],
                    mode='lines',
                    line=dict(color='red', dash = 'dash'),
                    name='+5%'
                    )
        )
        fig_scatter.add_trace(go.Scattergl(
                    y=dfplot['lim_inf'], 
                    x=dfplot[time_col],
                    mode='lines',
                    line=dict(color='red', dash = 'dash'),
                    name='-5%'
                    )
        )
        fig_scatter.update_xaxes(title_text="Data")
        fig_scatter.update_yaxes(title_text= "Erro Médio Percentual", showgrid=False, zerolinecolor='#000000')
        fig_scatter = format_fig(fig_scatter, '', x_title=time_col, y_title='Erro Médio Percentual')
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with st.expander(f'Visualização do volume previsto e erros: {data_group} da ' + st.session_state['chosen_item']):
        
        fig_group_1 = go.Figure()
        fig_group_2 = go.Figure()
        
        sd = st.selectbox(f'{data_group}', group_items)

        dfplot2 = df.loc[df[data_group] == sd].copy()
        
        dfplot2['lim_sup'] = 5
        dfplot2['lim_inf'] = -1*dfplot2['lim_sup']
        
        col_group = st.columns(2)
        
        if 'nome' in dfplot2.columns:
            data_group_name = dfplot2['nome'].unique().tolist()[0]
        else:
            data_group_name = sd
            
        col_group[0].metric(label=data_group,
                value= data_group_name,
                delta=f"")

        days_count = dfplot2.shape[0]
        col_group[1].metric(label="Período",
                value=f"{days_count} dias")   
    
        fig_group_1.add_trace(go.Scattergl(x= dfplot2[time_col],
                                        y= dfplot2[y_true],
                                        mode='lines',
                                        line=dict(color='rgb(0, 0, 0)', dash = 'dash'),
                                        name='Real'
                                        )
        )
        
        for num, prev in enumerate(y_benchmark):
            
            dfplot2['residuo'] = dfplot2[prev] - dfplot2[y_true]
            dfplot2['mpe'] = MPE(dfplot2[y_true], dfplot2[prev])
            dfplot2['mpe'] = dfplot2['mpe'].clip(-100, 100)
            dfplot2['mape'] = np.abs(dfplot2['mpe'])
            dfplot2['acima5'] = np.where(dfplot2['mape']>5, 1, 0)
            dfplot2['acima20'] = np.where(dfplot2['mape']>20, 1, 0)
            
            #days_count = dfplot2.shape[0]
            
            mape_metrica = dfplot2.mape.mean()
            
            acima5_mask = (dfplot2['acima5']==True)
            days_acima5 = dfplot2.loc[acima5_mask].shape[0]
            perc_acima5 = days_acima5/days_count
            
            acima20_mask = (dfplot2['acima20']==True)
            days_acima20 = dfplot2.loc[acima20_mask].shape[0] 
            perc_acima20 = days_acima20/days_count

            colg = st.columns(4)
            deltag = np.round(mape_metrica-5,2)
            
            #col2[0].metric(label=data_group,value= st.session_state['chosen_item'],delta=f"")
            #col2[1].metric(label="Período", value=f"{days_count} dias")
            colg[0].metric(label="Previsto", value=f"{prev}")
            
            colg[1].metric(label="MAPE",
                        value=f"{round(mape_metrica,2)}%",
                        delta=f"{deltag}%",
                        delta_color="inverse")
            
            colg[2].metric(label="Dias Acima de 5%",
                        value=f"{round(100*perc_acima5,2)}%",
                        delta=f"{days_acima5} dias",
                        delta_color='off')
            
            colg[3].metric(label="Dias Acima de 20%",
                        value=f"{round(100*perc_acima20,2)}%",
                        delta=f"{days_acima20} dias",
                        delta_color='off')

            fig_group_1.add_trace(go.Scattergl(x= dfplot2[time_col],
                                        y= dfplot2[prev],
                                        mode='lines+markers',
                                        line=dict(color=rgb_list[num]),
                                        name=prev
                                        )
            )
            
            fig_group_2.add_trace(go.Scattergl(x=dfplot2[time_col],
                                y=dfplot2['mpe'],
                                mode='markers',
                                line=dict(color=rgb_list[num]),
                                name= prev
                                )
            )
            
        fig_group_1.update_xaxes(title_text="Data")
        fig_group_1.update_yaxes(title_text= "Erro Médio Percentual", showgrid=False, zerolinecolor='#000000')
        fig_group_1 = format_fig(fig_group_1, '', x_title=time_col, y_title='Volume')
        st.plotly_chart(fig_group_1, use_container_width=True)
        
        # TESTE 
        fig_group_2.add_trace(go.Scattergl(
                    y=dfplot2['lim_sup'], 
                    x=dfplot2[time_col],
                    mode='lines',
                    line=dict(color='red', dash = 'dash'),
                    name='+5%'
                    )
        )
        fig_group_2.add_trace(go.Scattergl(
                    y=dfplot2['lim_inf'], 
                    x=dfplot2[time_col],
                    mode='lines',
                    line=dict(color='red', dash = 'dash'),
                    name='-5%'
                    )
        )
        fig_group_2.update_xaxes(title_text="Data")
        fig_group_2.update_yaxes(title_text= "Erro Médio Percentual", showgrid=False, zerolinecolor='#000000')
        fig_group_2 = format_fig(fig_group_2, '', x_title=time_col, y_title='Erro Médio Percentual')
        st.plotly_chart(fig_group_2, use_container_width=True)
    
    def colorize_mape(cell_value):
        
        erro_acima = 'background-color: lightcoral;'
        erro_abaixo = 'background-color: yellow;'
        default = 'background-color: lightgreen;'
        null_cell = 'background-color: lightgray;'
        
        if type(cell_value) in [float, int]:
            if cell_value>5:
                return erro_acima 
            if cell_value<-5:
                return erro_abaixo 
    
        return default
    
    tbs = st.checkbox('Carregar Tabelas')
    if tbs:
        with st.spinner('Um instante...'):

            erro_cols = []
            
            for col in y_benchmark:
                df[f'erro_{col}'] = 100*(df[col] - df[y_true])/df[col]
                erro_cols.append(f'erro_{col}')
            
            #st.write(f'Tabela - {data_group}')
            st.dataframe(df.loc[df[data_group]!='CG24',[time_col, data_group, classe, y_true]+erro_cols+y_benchmark].
                        style.applymap(colorize_mape, subset=erro_cols))
            
            erro_cols = []
            
            for col in y_benchmark:
                benchmark_df[f'erro_{col}'] = 100*(benchmark_df[col] - benchmark_df[y_true])/benchmark_df[col]
                erro_cols.append(f'erro_{col}')
            
            #st.write(f'Tabela - {classe}')
            st.dataframe(benchmark_df[[time_col, classe, y_true]+erro_cols+y_benchmark].
                        style.applymap(colorize_mape, subset=erro_cols)
            )
    
def open_page(data:pd.DataFrame,
              time_col:str,
              data_group:str,
              classe:str,
              y_true:str,
              y_benchmark:str):
    
    try:
        st.subheader(f'Comparação dos agrupamentos')
        create_benchmark_view(data, time_col, data_group, classe,y_true, y_benchmark)
    except:
        st.warning('Carregue o arquivo em ''Leitura de Arquivos'' na aba lateral')
        st.stop()
        