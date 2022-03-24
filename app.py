from time import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go # 

from dashboard import *
from utils import load_data, preprocess_dataframe

from pages.page1 import create_global_metrics
from pages.page2 import create_grouped_radar
from pages.page3 import check_residuals, check_mape, plot_seasonal_decompose, plot_series, standard_residual, check_holidays

os.environ['TZ'] = 'UTC'
MENU = ['Métricas Globais',
        'Agrupamentos',
        'Análise de Resíduos',
        'Benchmark']
        
def main():
    st.sidebar.title("Navegação")
    choice = st.sidebar.radio(
     "", MENU)

    # Resume to iterator
    if 'file_path' not in st.session_state:
        st.session_state['file_path'] = None
            
    if 'id' not in st.session_state:
        st.session_state['id'] = None  
        
    if 'time_col' not in st.session_state:
        st.session_state['time_col'] = None
        
    if 'real' not in st.session_state:
        st.session_state['real'] = None
        
    if 'previsto' not in st.session_state:
        st.session_state['previsto'] = None
        
    if 'classes' not in st.session_state:
        st.session_state['classes'] = None
        
    if 'agrupamento' not in st.session_state:
        st.session_state['agrupamento'] = None
        
    if 'chosengroup' not in st.session_state:
        st.session_state['chosengroup'] = None
    
    if 'selected' not in st.session_state:
        st.session_state['selected'] = None
    
    if 'chosen_col' not in st.session_state:
        st.session_state['chosen_col'] = None
    
    if 'chosen_item' not in st.session_state:
        st.session_state['chosen_item'] = None
            
    if 'df' not in st.session_state:
        st.session_state['df'] = None
        
    if 'updated_df' not in st.session_state:
        st.session_state['updated_df'] = None
        
    if 'grouped_df' not in st.session_state:
        st.session_state['grouped_df'] = None
    
    with st.sidebar.expander("Leitura de arquivo"):    
        st.markdown('### Carregue o arquivo CSV 👇')

        file = st.file_uploader("",type=["csv"], key = 'uploader')
        
        if file is not None:
            st.session_state['file_path'] = file.name
            file_details = {"nome do arquivo":st.session_state['file_path'],
                    "tipo do arquivo":file.type}
            
            st.session_state['df'] = load_data(file)
            st.session_state['id'] = st.selectbox("Identificador:", st.session_state['df'].columns)
            st.session_state['time_col'] = st.selectbox("Coluna Temporal:", st.session_state['df'].columns)
            
            # TROQUEI O PREVISTO PELO REAL
            st.session_state['real'] = st.selectbox("Real (QDR):", st.session_state['df'].columns)
            st.session_state['previsto'] = st.selectbox("Previsto (QDP):", st.session_state['df'].columns)
            st.session_state['classes'] = st.multiselect("Classes:", st.session_state['df'].columns)
            st.session_state['agrupamento'] = st.selectbox("Agrupamento:",['NÃO']+st.session_state['df'].columns.tolist())
            st.session_state['df']['NÃO'] = 0
            
            data_group = st.session_state['id']
            time_col = st.session_state['time_col']
            y_true  = st.session_state['real']
            y_predicted = st.session_state['previsto']
            classes = st.session_state['classes']
            data_group2 = st.session_state['agrupamento']
            
            st.session_state['grouped_df'] = st.session_state['df'][[data_group,
                                                                     data_group2,
                                                                     time_col,
                                                                     y_true,
                                                                     y_predicted]]
            
            st.session_state['chosengroup'] = st.selectbox(f"Selecione o agrupamento:",
                            sorted(st.session_state['df'][data_group2].unique().tolist()))
            chosen_group = st.session_state['chosengroup']
            
            st.session_state['df'] = st.session_state['df'][st.session_state['df'][data_group2]==chosen_group]  
                
            try:
                st.session_state['df'] = preprocess_dataframe(st.session_state['df'],
                                                              time_col,
                                                              y_true,
                                                              y_predicted)
                
            except:
                pass
        else:
            st.warning('Carregue arquivo')
            
    try:
        st.subheader('Intervalo:')
        start_date, end_date = st.slider('',
                            value=[st.session_state['df'][time_col].min(), st.session_state['df'][time_col].max()],
                            max_value = st.session_state['df'][time_col].max(),
                            min_value = st.session_state['df'][time_col].min(),
                            key='first')
        
        if start_date <= end_date:
            pass
        else:
            st.warning('Error: Fim < Inicio.')

        st.write('Período:', start_date, '-', end_date)
        mask = (st.session_state['df'][time_col] >= start_date) & (st.session_state['df'][time_col] <= end_date)
        st.session_state['df'] = st.session_state['df'].loc[mask]
    
        #try: df = preprocess_dataframe(df,time_col,y_true,y_predicted)
        #except: pass
        st.session_state['updated_df'] = st.session_state['df'].copy()
    
    except:
        pass
    
    with st.expander("Dados"):
        try:
            st.dataframe(st.session_state['updated_df'][[data_group,
                                data_group2,
                                time_col,
                                y_true,
                                y_predicted]+classes])
        except:
            st.warning("Sem arquivo")
                
    ########################################## TELA 1 ##########################################
    if choice == 'Métricas Globais':
        try:
            st.subheader(f'Métricas para o agrupamento: {chosen_group}')
            
            create_global_metrics(st.session_state['updated_df'],
                                  time_col,
                                  data_group,
                                  classes,
                                  y_true,
                                  y_predicted)   
        except:
            st.warning('Carregue o arquivo em ''Leitura de Arquivos'' na aba lateral')

    ########################################## TELA 2 ##########################################
    elif choice == 'Agrupamentos':
        st.subheader(f'Comparação dos agrupamentos')
        try:
            create_grouped_radar(st.session_state['grouped_df'],
                                 data_group,
                                 data_group2,
                                 time_col,
                                 y_true,
                                 y_predicted) 
        except:
            st.warning('Carregue o arquivo em ''Leitura de Arquivos'' na aba lateral')
            
    ########################################## TELA 3 ##########################################
    elif choice == 'Análise de Resíduos': 
          
        try:
            st.session_state['selected'] = st.selectbox(f"Selecione o {data_group}:",
                                    sorted(st.session_state['updated_df'][data_group].unique().tolist()))

            days_count = st.session_state['updated_df'][st.session_state['updated_df'][data_group]==st.session_state['selected']].shape[0]
            
            mape_metrica = st.session_state['updated_df'][st.session_state['updated_df'][data_group]==st.session_state['selected']].mape.clip(0,100).mean()
            
            acima5_mask = (st.session_state['updated_df']['acima5']==True) & \
                (st.session_state['updated_df'][data_group]==st.session_state['selected'])
                
            days_acima5 = st.session_state['updated_df'].loc[acima5_mask].shape[0]
            perc_acima5 = days_acima5/days_count
            
            acima20_mask = (st.session_state['updated_df']['acima20']==True) & \
                (st.session_state['updated_df'][data_group]==st.session_state['selected'])
            
            days_acima20 = st.session_state['updated_df'].loc[acima20_mask].shape[0] 
            perc_acima20 = days_acima20/days_count
        
            col1 = st.columns(5)
            delta1 = np.round(mape_metrica-5,2)

            col1[0].metric(label=data_group,
                        value= str(st.session_state['selected']),
                        delta=f"")
            col1[1].metric(label="Período",
                        value=f"{days_count} dias")
            col1[2].metric(label="MAPE",
                        value=f"{round(mape_metrica,2)}%",
                        delta=f"{delta1}%",
                        delta_color="inverse")
            col1[3].metric(label="Dias Acima de 5%",
                        value=f"{round(100*perc_acima5,2)}%",
                        delta=f"{days_acima5} dias",
                        delta_color='off')
            col1[4].metric(label="Dias Acima de 20%",
                        value=f"{round(100*perc_acima20,2)}%",
                        delta=f"{days_acima20} dias",
                        delta_color='off')
            
            plot_series(st.session_state['updated_df'],
                    st.session_state['time_col'] ,
                    st.session_state['real'] ,
                    st.session_state['previsto'] ,
                    st.session_state['id'] ,
                    st.session_state['selected'])
        except:
            pass
        
        with st.expander('Decomposição Clássica'):
            try:
                chosen = st.selectbox('',  sorted(st.session_state['updated_df'].columns.tolist()))
                
                plot_seasonal_decompose(st.session_state['updated_df'],
                                        data_group,
                                        st.session_state['selected'],
                                        time_col,
                                        col = chosen)
            except:
                st.warning('Selecione uma coluna numérica')
        
        try:   
            st.session_state['updated_df'] = standard_residual(st.session_state['updated_df'], data_group, y_true, y_predicted)
        except: 
            st.warning('não foi possível calcular o resíduo padronizado para esse conjunto de dados')
            
        try:   
            st.subheader("Resíduos")
            check_residuals(st.session_state['updated_df'],
                    time_col,
                    st.session_state['selected'],
                    data_group
                ) 
            check_mape(st.session_state['updated_df'],
                    time_col,
                    st.session_state['selected'],
                    data_group
                ) 
            
        except:
            st.warning('há um erro na parametrização dos dados, recarregue ou ajuste na *Aba de Navegação*')
        
        # CHECKING
        check_holidays(st.session_state['updated_df'],
                    time_col,
                    data_group
            )
        #check_rmse(df,time_col,selected,data_group) 
        
    ########################################## TELA 4 ##########################################
        
    elif choice == 'Benchmark':
        
        try:    
            st.session_state['chosen_col'] = st.selectbox('Categoria', classes)
            
            benchmark_df = st.session_state['updated_df']
            benchmark_df[st.session_state['time_col']] = pd.to_datetime(benchmark_df[st.session_state['time_col']])
            benchmark_df = benchmark_df.groupby([pd.Grouper(key = st.session_state['time_col'], freq = 'D'), st.session_state['chosen_col'] ]).sum().reset_index()
            benchmark_df = benchmark_df.reset_index()
            benchmark_df['residuo'] = benchmark_df[st.session_state['previsto']] - benchmark_df[st.session_state['real']]
            benchmark_df['mpe'] = 100*(benchmark_df['residuo']/benchmark_df[st.session_state['previsto']])
            benchmark_df['mape'] = np.abs(benchmark_df['mpe'])
            benchmark_df['acima5'] = np.where(benchmark_df['mape']>5, 1, 0)
            benchmark_df['acima20'] = np.where(benchmark_df['mape']>20, 1, 0)

            st.session_state['chosen_item'] = st.selectbox('Classe', benchmark_df[st.session_state['chosen_col']].unique().tolist())
            
            dfplot = benchmark_df.loc[benchmark_df[st.session_state['chosen_col'] ] == st.session_state['chosen_item']]
        
            days_count = dfplot.shape[0]
            mape_metrica = dfplot.mape.clip(0,100).mean()
            acima5_mask = (dfplot['acima5']==True)
            days_acima5 = dfplot.loc[acima5_mask].shape[0]
            perc_acima5 = days_acima5/days_count
            
            acima20_mask = (dfplot['acima20']==True)
            days_acima20 = dfplot.loc[acima20_mask].shape[0] 
            perc_acima20 = days_acima20/days_count
            
            col2 = st.columns(5)
            delta1 = np.round(mape_metrica-5,2)

            col2[0].metric(label=data_group,
                        value= st.session_state['chosen_item'],
                        delta=f"")
            
            col2[1].metric(label="Período",
                        value=f"{days_count} dias")
            
            col2[2].metric(label="MAPE",
                        value=f"{round(mape_metrica,2)}%",
                        delta=f"{delta1}%",
                        delta_color="inverse")
            
            col2[3].metric(label="Dias Acima de 5%",
                        value=f"{round(100*perc_acima5,2)}%",
                        delta=f"{days_acima5} dias",
                        delta_color='off')
            
            col2[4].metric(label="Dias Acima de 20%",
                        value=f"{round(100*perc_acima20,2)}%",
                        delta=f"{days_acima20} dias",
                        delta_color='off')
            
            fig = go.Figure()
            fig.add_trace(go.Scattergl(x= dfplot[st.session_state['time_col']],
                                        y= dfplot[st.session_state['real']],
                                        mode='lines',
                                        line=dict(color='rgb(32,4,114)'),
                                        name='Real'
                                        )
                )
            fig.add_trace(go.Scattergl(x= dfplot[st.session_state['time_col']],
                                        y= dfplot[st.session_state['previsto']],
                                        mode='lines',
                                        line=dict(color='rgb(234, 82, 111)'),
                                        name='Previsto'
                                        )
                )
            fig.update_xaxes(title_text="Data")
            fig.update_yaxes(title_text= "Residuo", showgrid=False, zerolinecolor='#000000')
            fig = format_fig(fig, '', x_title=time_col, y_title='Resíduo')
            st.plotly_chart(fig, use_container_width=True)

            with st.expander('tabela'):
                st.dataframe(dfplot[[st.session_state['time_col'],
                                    st.session_state['chosen_col'],
                                    st.session_state['real'],
                                    st.session_state['previsto'],
                                    'mpe',
                                    'mape',
                                    'acima5',
                                    'acima20'
                                    ]])
            fig = go.Figure()
            dfplot['lim_sup'] = 5
            dfplot['lim_inf'] = -1*dfplot['lim_sup']
            fig.add_trace(go.Scattergl(x=dfplot[time_col],
                                        y=dfplot['mpe'],
                                        mode='markers',
                                        line=dict(color='rgb(32,4,114)'),
                                        name='MPE'
                                        )
            )
            fig.add_trace(go.Scattergl(
                        y=dfplot['lim_sup'], 
                        x=dfplot[time_col],
                        line=dict(color='red', dash = 'dash'),
                        name='+5%'
                        )
            )
            fig.add_trace(go.Scattergl(
                        y=dfplot['lim_inf'], 
                        x=dfplot[time_col],
                        line=dict(color='red', dash = 'dash'),
                        name='-5%'
                        )
            )
            fig.update_xaxes(title_text="Data")
            fig.update_yaxes(title_text= "Erro Médio Percentual", showgrid=False, zerolinecolor='#000000')
            fig = format_fig(fig, '', x_title=time_col, y_title='Erro Médio Percentual')
            st.plotly_chart(fig, use_container_width=True)
        except:
            pass
        
if __name__ == "__main__":
    set_streamlit()
    set_page_container_style()
    main()
