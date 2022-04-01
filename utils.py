import streamlit as st
from google.oauth2 import service_account
from google.auth import Credentials
from gsheetsdb.db import Connection
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
def run_query(query, conn):
    rows = conn.execute(query, headers=1)
    rows = rows.fetchall()
    print(rows[-1])
    return rows
    
def load_sql_data():
    # Create a connection object.
    credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
            ],
        )
    sheet_url = os.environ["gsheets_url"]
    query_msg = f'SELECT * FROM "{sheet_url}"'
    
    @st.cache(allow_output_mutation=True, ttl=600, hash_funcs={Credentials:hash})
    def get_database_connection(credentials):
        return Connection(credentials=credentials)
    
    conn = get_database_connection(credentials)
    
    # Perform SQL query on the Google Sheet.
    # Uses st.cache to only rerun when the query changes or after 10 min.
    @st.cache(allow_output_mutation=True, ttl=600)
    def run_dataframe_query(query):
        return pd.read_sql(query, conn)

    return run_dataframe_query(query_msg)

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

def validate_password(password):
    return st.session_state['password'] == os.environ['app_password']
