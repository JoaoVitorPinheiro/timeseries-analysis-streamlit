import streamlit as st
from kpi import *

MESES =["Janeiro", "Fevereiro", "Março", "Abril",
          "Maio", "Junho", "Julho", "Agosto",
          "Setembro", "Outubro", "Novembro", "Dezembro"]

WEEKDAY = ["Segunda", "Terça",
           "Quarta","Quinta",
           "Sexta", "Sábado", "Domingo"]

nomear_dia = lambda x: WEEKDAY[x]
nomear_mes = lambda x: MESES[x-1]

@st.cache(allow_output_mutation=True)
def load_data(file):
    df = pd.read_csv(file, parse_dates=True)
    return df
            
def preprocess_dataframe(data: pd.DataFrame,
                         time_col: str,
                         y_true: str,
                         y_predicted: str,
                         ) -> pd.DataFrame:
    
    data[time_col] = pd.to_datetime(data[time_col], format = '%Y-%m-%d')
    data[time_col] = data[time_col].dt.date
    nan_mask = (data[y_true].isna())|(data[y_predicted].isna())
    data = data[~nan_mask]   # remove some nan's only
    #Clip para previsões negativas
    data[y_predicted] = np.where(data[y_predicted]<0, 0, data[y_predicted])
    data['mape'] = MAPE(data[y_true],data[y_predicted])
    data['rmse'] = RMSE(data[y_true],data[y_predicted])

    #Limiar do MAPE para evitar distorções
    data['mape'] = np.where(data['mape']>100, 100, data['mape'])

    data['mpe'] = MPE(data[y_true],data[y_predicted])
    data['residuo'] = data[y_true] - data[y_predicted]
    data['acima5'] = np.where(data['mape']>5, True, False)
    data['acima20'] = np.where(data['mape']>20, True, False)
    data[y_true+'_diff'] = data[y_true].diff()
    data = data.sort_values(by = time_col, ascending=True)
    return data
