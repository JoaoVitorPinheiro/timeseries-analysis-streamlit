import numpy as np
import pandas as pd

from dashboard import *
from utils import load_data, preprocess_dataframe

from pages.page1 import create_global_metrics
from pages.page2 import create_grouped_radar
from pages.page3 import check_residuals, check_mape, plot_seasonal_decompose, plot_series, standard_residual, check_holidays

os.environ['TZ'] = 'UTC'
MENU = ['Métricas Globais', 'Agrupamentos', 'Análise de Resíduos',
        #'Benchmark'
        ]
        
def main():
    st.sidebar.title("Navegação")
    choice = st.sidebar.radio(
     "", MENU)

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
    
    if 'updated_df' not in st.session_state:
        st.session_state['file_path'] = None
        
    if 'grouped_df' not in st.session_state:
        st.session_state['file_path'] = None
        
    with st.sidebar.expander("Leitura de arquivo"):    
        st.markdown('### Carregue o arquivo CSV 👇')

        file = st.file_uploader("",type=["csv"], key = 'uploader')
        
        if file is not None:
            st.session_state['file_path'] = file.name
            file_details = {"nome do arquivo":st.session_state['file_path'],
                    "tipo do arquivo":file.type}
            
            #'Dados', file_details
            
            df = load_data(file)
            #with st.expander("Informações dos dados:"):
            #    st.write(file_details)

            st.session_state['id'] = st.selectbox("Identificador:", df.columns)
            st.session_state['time_col'] = st.selectbox("Coluna Temporal:", df.columns)
            # TROQUEI O PREVISTO PELO REAL
            st.session_state['real'] = st.selectbox("Real (QDR):", df.columns)
            st.session_state['previsto'] = st.selectbox("Previsto (QDP):", df.columns)
            st.session_state['classes'] = st.multiselect("Classes:", df.columns)
            st.session_state['agrupamento'] = st.selectbox("Agrupamento:",['NÃO']+df.columns.tolist())
            
            df['NÃO'] = 0
            
            data_group = st.session_state['id']
            time_col = st.session_state['time_col']
            y_true  = st.session_state['real']
            y_predicted = st.session_state['previsto']
            classes = st.session_state['classes']
            data_group2 = st.session_state['agrupamento']
            
            st.session_state['grouped_df'] = df[[data_group,
                                                 data_group2,
                                                 time_col,
                                                 y_true,
                                                 y_predicted
                                                 ]]
            
            st.session_state['chosengroup'] = st.selectbox(f"Selecione o agrupamento:",
                            sorted(df[data_group2].unique().tolist()))
            chosen_group = st.session_state['chosengroup']
            
            df = df[df[data_group2]==chosen_group]  
                
            try:
                df = preprocess_dataframe(df,
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
                            value=[df[time_col].min(), df[time_col].max()],
                            max_value = df[time_col].max(),
                            min_value = df[time_col].min(),
                            key='first')
        
        if start_date <= end_date:
            pass
        else:
            st.warning('Error: Fim < Inicio.')

        st.write('Período:', start_date, '-', end_date)
        mask = (df[time_col] >= start_date) & (df[time_col] <= end_date)
        df = df.loc[mask]
    
        #try: df = preprocess_dataframe(df,time_col,y_true,y_predicted)
        #except: pass
        st.session_state['updated_df'] = df.copy()
    
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
                chosen = st.selectbox('',  sorted(df.columns.tolist()))
                
                plot_seasonal_decompose(st.session_state['updated_df'],
                                        data_group,
                                        st.session_state['selected'],
                                        time_col,
                                        col = chosen)
            except:
                st.warning('Selecione uma coluna numérica')
        
        try:   
            df = standard_residual(st.session_state['updated_df'], data_group, y_true, y_predicted)
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
    #elif choice == 'Benchmark':
    # Recebe o modelo 2
    # Abre uma janela para leitura de dados
    #    pass     
if __name__ == "__main__":
    set_streamlit()
    set_page_container_style()
    main()
