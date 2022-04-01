from time import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go 

from dashboard import *
from utils import load_csv_data, load_sql_data, preprocess_dataframe, validate_password

from pages.page1 import create_global_metrics
from pages.page2 import create_grouped_radar
from pages.page3 import check_residuals, check_mape, plot_seasonal_decompose, plot_series, standard_residual, check_holidays
from pages.page4 import create_benchmark_view

os.environ['TZ'] = 'UTC'

MENU = ['Métricas Globais',
        #'Agrupamentos',
        'Análise de Resíduos',
        'Benchmark']
     
def main():
    
    set_streamlit()
    set_page_style()
    # Resumir st.session_state em um iterator
    st_vars = []
    
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

    if 'previsto_compare' not in st.session_state:
        st.session_state['previsto_compare'] = None
        
    if 'classe' not in st.session_state:
        st.session_state['classe'] = None
        
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
    
    if 'navigator' not in st.session_state:
        st.session_state['navigator'] = None
    
    if 'password' not in st.session_state:
        st.session_state['password'] = None
              
    st.sidebar.title("Páginas")
    st.session_state['navigator'] = st.sidebar.radio(
     "", MENU)
    choice = st.session_state['navigator']
    
    st.sidebar.title("Carregamento de dados")
    file_menu = st.sidebar.radio("",('Arquivo CSV', 'Teste'))

    if file_menu == 'Teste':
        
        st.session_state['password']  = st.sidebar.text_input(label='', type = 'password')

        #if st.session_state['password'] != os.environ['app_password']: 
        if validate_password(st.session_state['password']) == False: 
            st.warning('Acesso negado')
            st.stop()
            
        st.session_state['df'] = load_sql_data()
        st.session_state['file_path'] = "gsheets"
        #st.dataframe(st.session_state['df'])
        st.session_state['id'] = 'CityGate'
        st.session_state['time_col'] = 'Data'
        st.session_state['real'] = 'QDR Comgás'
        st.session_state['previsto'] = 'Comgás'
        st.session_state['previsto_compare'] = ['Comgás','LGBM_PCS','Extrap_PCS']
        st.session_state['classe'] = 'Zona_Entrega'
        st.session_state['agrupamento'] = 'NÃO'
        st.session_state['df']['NÃO'] = 0
        
        data_group = st.session_state['id']
        time_col = st.session_state['time_col']
        y_true  = st.session_state['real']
        y_predicted = st.session_state['previsto']
        y_benchmark = st.session_state['previsto_compare']
        classe = st.session_state['classe']
        data_group2 = st.session_state['agrupamento']
        
        st.session_state['grouped_df'] = st.session_state['df'][[data_group,
                                                                data_group2,
                                                                time_col,
                                                                y_true,
                                                                y_predicted]]
        
        st.session_state['chosengroup'] = 0
        chosen_group = st.session_state['chosengroup']
        
        st.session_state['df'] = st.session_state['df'][st.session_state['df'][data_group2]==chosen_group]  
            
        #try:
        st.session_state['df'] = preprocess_dataframe(st.session_state['df'],
                                                    time_col,
                                                    y_true,
                                                    y_predicted,
                                                    y_benchmark)
            
        #except: pass
        
    else:
        with st.sidebar.expander("Carregamento de arquivo"):    
            
            st.markdown('### Carregue o arquivo CSV 👇')

            file = st.file_uploader("",type=["csv"], key = 'uploader')
            
            if file is not None:
                st.session_state['file_path'] = 'csv'
                file_details = {"nome do arquivo":st.session_state['file_path'],
                        "tipo do arquivo":file.type}
                
                try:
                    st.session_state['df'] = load_csv_data(file)
                except:
                    st.warning('Erro no carregamento')
                    st.stop()
                    
                st.session_state['id'] = st.selectbox("Identificador:", st.session_state['df'].columns)
                st.session_state['time_col'] = st.selectbox("Coluna Temporal:", st.session_state['df'].columns)
                
                # TROQUEI O PREVISTO PELO REAL
                st.session_state['real'] = st.selectbox("Real :", st.session_state['df'].columns)
                st.session_state['previsto'] = st.selectbox("Previsto :", st.session_state['df'].columns)
                #st.session_state['previsto_compare'] = st.selectbox("Previsto (Benchmark):", st.session_state['df'].columns)
                st.session_state['previsto_compare'] = st.multiselect("Previsto (Benchmark):", st.session_state['df'].columns)
                st.session_state['classe'] = st.selectbox("Classe:", st.session_state['df'].columns)
                st.session_state['agrupamento'] = st.selectbox("Agrupamento:",['NÃO']+st.session_state['df'].columns.tolist())
                st.session_state['df']['NÃO'] = 0
                
                data_group = st.session_state['id']
                time_col = st.session_state['time_col']
                y_true  = st.session_state['real']
                y_predicted = st.session_state['previsto']
                y_benchmark = st.session_state['previsto_compare']
                classe = st.session_state['classe']
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
                    

                st.session_state['df'] = preprocess_dataframe(st.session_state['df'],
                                                                time_col,
                                                                y_true,
                                                                y_predicted,
                                                                y_benchmark)
                    
            else:
                st.warning('Selecione o arquivo e preencha os campos')
                st.stop()
    
    if st.session_state['file_path']:
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
            
            periodo = (end_date - start_date).days + 1
            
            time_dict = {'Início:':str(start_date),
                         'Fim:':str(end_date),
                         'Período:':str(periodo)+' dias'}
            
            st.write(time_dict)
            
            mask = (st.session_state['df'][time_col] >= start_date) & (st.session_state['df'][time_col] <= end_date)
            st.session_state['df'] = st.session_state['df'].loc[mask]
            st.session_state['updated_df'] = st.session_state['df'].copy()
        
        except:
            st.warning('Falha na data')
            st.stop()
        
        with st.expander("Dados"):
        
            st.dataframe(st.session_state['updated_df'][[data_group,
                                data_group2,
                                time_col,
                                y_true,
                                y_predicted]+[classe]])

                    
        ########################################## TELA 1 ##########################################
        if choice == 'Métricas Globais':
            try:
                st.subheader(f'Métricas para o agrupamento: {chosen_group}')
                
                create_global_metrics(st.session_state['updated_df'],
                                    time_col,
                                    data_group,
                                    [classe],
                                    y_true,
                                    y_predicted)   
            except:
                st.warning('Carregue o arquivo em ''Leitura de Arquivos'' na aba lateral')
                st.stop()

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
                st.stop()
        ########################################## TELA 3 ##########################################
        elif choice == 'Análise de Resíduos': 
            
            try:
                st.session_state['selected'] = st.selectbox(f"Selecione o {data_group}:",
                                        sorted(st.session_state['updated_df'][data_group].unique().tolist()))

                df_res = st.session_state['updated_df'][st.session_state['updated_df'][data_group]==st.session_state['selected']].copy()
                selected_class = df_res[classe].unique().tolist()[0]
                selected_class
                days_count = df_res.shape[0]
                
                mape_metrica = df_res.mape.clip(0,100).mean()
                
                acima5_mask = (df_res['acima5']==True) & \
                    (df_res[data_group]==st.session_state['selected'])
                    
                days_acima5 = df_res.loc[acima5_mask].shape[0]
                perc_acima5 = days_acima5/days_count
                
                acima20_mask = (df_res['acima20']==True) & \
                    (df_res[data_group]==st.session_state['selected'])
                
                days_acima20 = df_res.loc[acima20_mask].shape[0] 
                perc_acima20 = days_acima20/days_count

                col1 = st.columns(5)
                delta1 = np.round(mape_metrica-5,2)

                col1[0].metric(label=data_group,
                            value= str(st.session_state['selected']),
                            delta=f"{selected_class}",
                            delta_color='off')
                
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
                st.warning('carregue o arquivo')
                st.stop()
            
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
                create_benchmark_view(st.session_state['updated_df'], time_col, data_group, classe,y_true, y_benchmark)
            except:pass
            
        #except: pass
        
if __name__ == "__main__":
    main()
