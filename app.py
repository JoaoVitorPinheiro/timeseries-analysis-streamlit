from analysis import *
from dashboard import *
import os

os.environ['TZ'] = 'UTC'
MENU = ['Métricas Globais', 'Agrupamentos', 'Análise de Resíduos', 'Benchmark']

def main():
    st.sidebar.title("Navegação")
    choice = st.sidebar.radio(
     "", MENU)

    #with st.expander("Sobre"):
    #    st.markdown("""
    #        ####
    #            Análise de performance de modelos de previsão de séries temporais
    #        ###### Features
    #            - MAPE e RMSE dos modelos
    #            - Histogramas e Boxplots dos resíduos
    #            - Função de Autocorrelação dos resíduos
    #        ######""",
    #unsafe_allow_html = True)

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
            
    with st.sidebar.expander("Leitura de arquivo"):    
        st.markdown('### Carregue o arquivo CSV 👇')
        
        @st.cache(allow_output_mutation=True)
        def load_data(file):
            df = pd.read_csv(file, parse_dates=True)
            #df.columns = ['url', 'redir']
            return df
        
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
            st.session_state['real'] = st.selectbox("Série Real:", df.columns)
            st.session_state['previsto'] = st.selectbox("Série Prevista:", df.columns)
            st.session_state['classes'] = st.multiselect("Classes:", df.columns)
            st.session_state['agrupamento'] = st.selectbox("Agrupamento:",['NÃO']+df.columns.tolist())
            
            df['NÃO'] = 0
            
            data_group = st.session_state['id']
            time_col = st.session_state['time_col']
            y_true  = st.session_state['real']
            y_predicted = st.session_state['previsto']
            classes = st.session_state['classes']
            data_group2 = st.session_state['agrupamento']
            
            grouped = df[[data_group,
                          data_group2,
                          time_col,
                          y_true,
                          y_predicted]]
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
            st.warning('Erro ao carregar arquivo')
            
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
    
        try:
            #df = df[df[data_group2]==chosen_group]
            df = preprocess_dataframe(df,
                                    time_col,
                                    y_true,
                                    y_predicted)
        except:
            pass
    
    except:
        pass
    
    ########################################## TELA 1 ##########################################
    if choice == 'Métricas Globais':
        with st.expander("Dados"):
            try:
                st.dataframe(df[[data_group,
                                 data_group2,
                                 time_col,
                                 y_true,
                                 y_predicted]+classes])
            except:
                st.warning("Sem arquivo")
        try:
            st.subheader(f'Métricas para o agrupamento: {chosen_group}')
            create_global_metrics(df,
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
            create_grouped_radar(grouped,
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
                                    sorted(df[data_group].unique().tolist()))
            selected = st.session_state['selected']
        except: 
            pass      
        try:
            metrica = df[df[data_group]==selected].mape.mean()
            p_mask = (df['acima5']==True) & (df[data_group]==selected)
            perc_acima5 = df.loc[p_mask].shape[0]/df[df[data_group]==selected].shape[0]
            col1, col2, col3 = st.columns(3)
            delta2 = np.round(metrica-5,2)
        
            col1.metric(label=data_group,
                        value=f"{selected}",
                        delta="")
            col2.metric(label="MAPE",
                        value=f"{round(metrica,2)}%",
                        delta=f"{delta2}%",
                        delta_color="inverse")
            col3.metric(label="Dias Acima de 5%",
                        value=f"{round(100*perc_acima5,2)}%",
                        delta="")
        
            plot_series(df,
                    st.session_state['time_col'] ,
                    st.session_state['real'] ,
                    st.session_state['previsto'] ,
                    st.session_state['id'] ,
                    selected)
        except:
            pass
        
        try:   
            df = standard_residual(df, data_group, y_true, y_predicted)
        except: 
            st.warning('não foi possível calcular o resíduo padronizado para esse conjunto de dados')
        try:   
            st.subheader("Propriedades dos Resíduos")
        
            check_residuals(df,
                    time_col,
                    selected,
                    data_group
                    )  
            check_seasonal_residuals(df,
                    time_col,
                    selected,
                    data_group
                    ) 
            check_holidays(df,
                    time_col,
                    selected,
                    data_group
                ) 

        except:
            st.warning('há um erro na parametrização dos dados, recarregue ou ajuste na *Aba de Navegação*')
    
    ########################################## TELA 4 ##########################################
    elif choice == 'Benchmark':
    # Recebe o modelo 2
    # Abre uma janela para leitura de dados
        pass     
if __name__ == "__main__":
    set_streamlit()
    set_page_container_style()
    main()
