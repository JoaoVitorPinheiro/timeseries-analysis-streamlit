import streamlit as st
from gsheetsdb import connect
import os

from kpi import *

MESES =["Janeiro", "Fevereiro", "Março", "Abril",
          "Maio", "Junho", "Julho", "Agosto",
          "Setembro", "Outubro", "Novembro", "Dezembro"]

WEEKDAY = ["Segunda", "Terça",
           "Quarta","Quinta",
           "Sexta", "Sábado", "Domingo"]

nomear_dia = lambda x: WEEKDAY[x]
nomear_mes = lambda x: MESES[x-1]
    
@st.cache(allow_output_mutation=True, persist = True)
def load_csv_data(file):
    try:
        return pd.read_csv(file, parse_dates=True)
    except:
        st.error(
            "This file can't be converted into a dataframe. Please import a csv file with a valid separator."
        )
        st.stop()

@st.cache(allow_output_mutation=True, ttl=600)
def load_sql_data():
        # Create a connection object.
    conn = connect()
    # Perform SQL query on the Google Sheet.
    # Uses st.cache to only rerun when the query changes or after 10 min.
    def run_query(query):
        rows = conn.execute(query, headers=1)
        rows = rows.fetchall()
        return rows

    sheet_url = os.environ["gsheets_url"]
    print(sheet_url)
    query_msg = f'SELECT * FROM "{sheet_url}"'
    rows = run_query(query_msg)
    print(rows[-1])
    df = pd.read_sql(query_msg, conn)
    return df

def preprocess_dataframe(data: pd.DataFrame,
                         time_col: str,
                         y_true: str,
                         y_predicted: str,
                         y_benchmark: str,
                         ) -> pd.DataFrame:
    
    if 'Nome' in data.columns:
        data['nome'] = data['Nome']
        
    data[time_col] = pd.to_datetime(data[time_col], format = '%Y-%m-%d')
    data[time_col] = data[time_col].dt.date
    nan_mask = (data[y_true].isna())|(data[y_predicted].isna())
    data = data[~nan_mask]   # remove nan's 
    
    data[y_predicted] = data[y_predicted].astype(float)
    data[y_true] = data[y_true].astype(float)
    for col in y_benchmark:
        data[col] = data[col].astype(float)

    data[y_predicted] = np.where(data[y_predicted]<0, 0, data[y_predicted])    #Clip para previsões negativas
    data['rmse'] = RMSE(data[y_true],data[y_predicted])
    data['mpe'] = MPE(data[y_true],data[y_predicted])
    data['mape'] = np.abs(data['mpe'])
    data['mape'] = np.where(data['mape']>100, 100, data['mape'])     #Limiar do MAPE para evitar distorções
    data['residuo'] = data[y_true] - data[y_predicted]
    data['acima5'] = (data['mape']>5).astype(int)
    data['acima20'] = (data['mape']>20).astype(int)
    data = data.sort_values(by = time_col, ascending=True)
    return data
