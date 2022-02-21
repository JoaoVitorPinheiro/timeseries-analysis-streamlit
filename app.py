from analysis import *
from dashboard import *
import os

os.environ['TZ'] = 'UTC'
MENU = ['Métricas Globais', 'Análise de Resíduos', 'Benchmark']

def main():
    st.sidebar.title("Navegação")
    choice = st.sidebar.radio(
     "", MENU)
    
    with st.expander("Sobre"):
        st.markdown("""
            ####
                Análise de performance de modelos de previsão de séries temporais
            ###### Features
                - MAPE e RMSE dos modelos
                - Histogramas e Boxplots dos resíduos
                - Função de Autocorrelação dos resíduos
            ######""",
    unsafe_allow_html = True)
    
    with st.sidebar.expander("Leitura de arquivo"):    
        data_file = st.file_uploader("Selecionar arquivo CSV",type=["csv"])
        if data_file is not None:
            file_details = {"nome do arquivo":data_file.name,
                    "tipo do arquivo":data_file.type,
                    "tamanho do arquivo":data_file.size}

            df = pd.read_csv(data_file, parse_dates=True)
            #with st.expander("Informações dos dados:"):
            #    st.write(file_details)
                
            data_group = st.selectbox("1° Grupo:", df.columns)
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
            except:
                pass
        
    try:
        st.subheader('Intervalo:')
        start_date, end_date = st.slider('',
                            value=[df[time_col].min(), df[time_col].max()],
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
    
    if choice == 'Métricas Globais':
        with st.expander("Dados"):
            try:
                st.dataframe(df)
            except:
                st.warning("Sem arquivo")
        try:
            create_global_metrics(df, data_group, y_true, y_predicted)   
        except:
            st.warning('Carregue o arquivo em ''Leitura de Arquivos'' na aba lateral')
            
    elif choice == 'Análise de Resíduos':    
        try:
            selected = st.selectbox(f"Selecione o {data_group}:",
                                    sorted(df[data_group].unique().tolist()))
        except: 
            pass      
        try:
            metrica = df[df[data_group]==selected].mape.mean()
            p_mask = (df['acima5']==True) & (df[data_group]==selected)
            perc_acima5 = df.loc[p_mask].shape[0]/df[df[data_group]==selected].shape[0]
            col1, col2, col3 = st.columns(3)
            delta2 = np.round(metrica-5,2)
            delta3 = perc_acima5-5
            
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
                    time_col,
                    y_true,
                    y_predicted,
                    data_group,
                    selected,
                    
                    period = 'D',
                    diff = 0)
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
        
    elif choice == 'Benchmark':
    # Recebe o modelo 2
    # Abre uma janela para leitura de dados
        pass     
if __name__ == "__main__":
    set_streamlit()
    set_page_container_style()
    main()
