import streamlit as st
from dashboard import *
from analysis import *

def create_initial_page(df, time_col, y_true, y_predicted):
    
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
    
    return df
