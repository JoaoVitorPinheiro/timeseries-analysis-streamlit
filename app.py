from cmath import e
from analysis import *
from dashboard import *

def main():
    st.sidebar.title("Navegação")
    choice = st.sidebar.radio(
     "",
     ('Métricas Globais', 'Análise de Resíduos', 'Documentary'))
    
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
                
            data_group = st.selectbox("Selecione o grupo:", df.columns)
            time_col = st.selectbox("Selecione a coluna temporal:", df.columns)
            y_true = st.selectbox("Selecione a série real:", df.columns)
            y_predicted = st.selectbox("Selecione a série prevista:", df.columns)
            
            df = preprocess_dataframe(df,
                                    data_group,
                                    time_col,
                                    y_true,
                                    y_predicted)
        
    try:
        st.sidebar.subheader('Recorte Temporal:')
        start_date, end_date = st.sidebar.slider('',
                            value=[df[time_col].min(), df[time_col].max()],
                            key='first')

        if start_date <= end_date:
            pass
        else:
            st.sidebar.warning('Error: Fim < Inicio.')

        st.sidebar.write('Período:', start_date, '-', end_date)
        mask = (df[time_col] >= start_date) & (df[time_col] <= end_date)
        df = df.loc[mask]
        df = preprocess_dataframe(df,
                                data_group,
                                time_col,
                                y_true,
                                y_predicted)
    except:
        pass
    
    if choice == 'Métricas Globais':
        with st.expander("Dados"):
            try:
                st.dataframe(df)
            except:
                st.warning("Sem arquivo")
        try:
            create_global_metrics(df, data_group)   
        except:
            st.warning('Carregue o arquivo em ''Leitura de Arquivos'' na aba lateral')
    
    elif choice == 'Análise de Resíduos':    
        try:
            #selected = st.selectbox(f"Selecione o {data_group}:",
            #                        sorted(df[data_group].unique().tolist()))'''
            selected = st.select_slider(
                f"Selecione o {data_group}:",
                options=sorted(df[data_group].unique().tolist()))
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
            col3.metric(label="Acima de 5%",
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
            st.header("2. Propriedades dos Resíduos")
            standardize = st.checkbox('Resíduo Padronizado')
            check_residuals(df,
                        time_col,
                        selected,
                        data_group,
                        standardize=standardize)
        except: 
            pass
        
if __name__ == "__main__":
    set_streamlit()
    set_page_container_style()
    main()
