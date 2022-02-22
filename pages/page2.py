import streamlit as st
from dashboard import *
from analysis import *

def create_page2(df, data_group, time_col, y_true, y_predicted):
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

