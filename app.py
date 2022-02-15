from analysis import *
from dashboard import *

def main():
    
    st.header("Análise da Previsão")

    st.sidebar.title('Setup')

    data_file = st.sidebar.file_uploader("Upload CSV",type=["csv"])
		
    if data_file is not None:
        file_details = {"nome do arquivo":data_file.name,
                  "tipo do arquivo":data_file.type,
                  "tamanho do arquivo":data_file.size}
        
        df = pd.read_csv(data_file)
        
        #with st.expander("Informações dos dados:"):
        #    st.write(file_details)
            
        data_group = st.sidebar.selectbox("Selecione o grupo:", df.columns)
        time_col = st.sidebar.selectbox("Selecione a coluna temporal:", df.columns)
        y_true = st.sidebar.selectbox("Selecione a série real:", df.columns)
        y_predicted = st.sidebar.selectbox("Selecione a série prevista:", df.columns)
        
        try:
            df = preprocess_dataframe(df, data_group, time_col, y_true, y_predicted)
            st.dataframe(df[data_group, time_col, y_true, y_predicted])
            
        except:
            pass
        
        with st.expander("Parâmetros:"):
            try:
                start_date, end_date = st.slider('',
                                                    df[time_col].min(),
                                                    df[time_col].max(),
                                                    value=[df[time_col].min(), df[time_col].max()],
                                                    key='first')
                
                if start_date < end_date:
                    pass
                else:
                    st.error('Error: Fim < Inicio.')

                st.write('Período selecionado:', start_date, '-', end_date)
                mask = (df[time_col] > start_date) & (df[time_col] <= end_date)
                df = df.loc[mask]
                
                selected = st.selectbox(f"Selecione o {data_group}:", sorted(df[data_group].unique().tolist()))
            except: 
                pass      
            
        with st.expander("Métricas Globais"):
            try:
                st.dataframe(df.groupby([data_group]).mape.mean().reset_index())
                st.dataframe(df)
                
            except:
                st.write('Falha')
        
        try:
            
            metrica = df[df[data_group]==selected].mape.mean()
            p_mask = (df['acima5']==True) & (df[data_group]==selected)
            perc_acima = df.loc[p_mask].shape[0]/df.shape[0]
            
            col1, col2, col3 = st.columns(3)
            #st.metric(label=data_group, value=f"{selected}", delta="")
            col1.metric(label=data_group, value=f"{selected}", delta="")
            col2.metric(label="MAPE", value=f"{round(metrica,2)}%", delta="")
            col3.metric(label="Acima de 5%", value=f"{round(100*perc_acima,2)}%", delta="")
            
            plot_series(df,
                    time_col,
                    y_true,
                    y_predicted,
                    data_group,
                    selected,
                    # style: Dict[Any, Any],
                    period = 'D',
                    diff = 0)
        except:
            pass
        
        try:
            plot_autocorrelation(df,
                        time_col,
                        y_true,
                        y_predicted,
                        selected,
                        data_group,
                        period = 'D',
                        diff = 0)
        except:
            pass
           
        try:
            check_residuals(df,
                        time_col,
                        y_true,
                        y_predicted,
                        selected,
                        data_group,
                        period = 'D',
                        diff = 0)
        except:
            pass
            
        st.subheader(": ")

if __name__ == "__main__":
    set_streamlit()
    set_page_container_style()
    
    main()
