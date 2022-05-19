import streamlit as st
from google.oauth2 import service_account
from gsheetsdb.db import Connection
import os

from kpi import *

MESES = ["Janeiro", "Fevereiro", "Março", "Abril",
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
    
    # Perform SQL query on the Google Sheet.
    # Uses st.cache to only rerun when the query changes or after 10 min.
    @st.cache(allow_output_mutation=True, ttl=600)
    def run_dataframe_query(query):
        
        conn = Connection()
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
    data['mpe'] = data['mpe'].clip(-100, 100) 
    data['mape'] = np.abs(data['mpe'])   #Limiar do MAPE para evitar distorções
    data['residuo'] = data[y_true] - data[y_predicted]
    data['acima5'] = (data['mape']>5).astype(int)
    data['acima20'] = (data['mape']>20).astype(int)
    data = data.sort_values(by = time_col, ascending=True)
    
    return data

def validate_password(password):
    return st.session_state['password'] == os.environ['app_password']

def init_file_upload():
    
    st.sidebar.title("Carregamento de dados")
    file_menu = st.sidebar.radio("",('Arquivo CSV', 'Teste'))

    if file_menu == 'Teste':
        
        st.session_state['password']  = st.sidebar.text_input(label='', type = 'password')

        if validate_password(st.session_state['password']) == False: 
            st.warning('Acesso negado')
            st.stop()
    
        st.session_state['df'] = load_sql_data()
        st.session_state['file_path'] = "gsheets"
        st.session_state['id'] = 'CityGate'
        st.session_state['time_col'] = 'Data'
        st.session_state['real'] = 'QDR Comgás'
        st.session_state['previsto'] = 'Comgás'
        st.session_state['previsto_compare'] = ['Comgás','LGBM_PCS','Extrap_PCS']
        st.session_state['classe'] = 'Zona_Entrega'
        st.session_state['agrupamento'] = 'NÃO'
        st.session_state['df']['NÃO'] = 0
        
        file_path = st.session_state['file_path']
        data_group = st.session_state['id']
        time_col = st.session_state['time_col']
        y_true  = st.session_state['real']
        y_predicted = st.session_state['previsto']
        y_benchmark = st.session_state['previsto_compare']
        classe = st.session_state['classe']
        data_group2 = st.session_state['agrupamento']
        grouped_df = st.session_state['df'][[data_group,data_group2,time_col,y_true,y_predicted]]
        st.session_state['chosen_group'] = 0
        chosen_group = st.session_state['chosen_group']
        
        df = st.session_state['df']#[st.session_state['df'][data_group2]==chosen_group]  
        df = preprocess_dataframe(df,time_col,y_true,y_predicted,y_benchmark)

    else:
        with st.sidebar.expander("Carregamento de arquivo"):    
            
            st.markdown('### Carregue o arquivo CSV 👇')
            file = st.file_uploader("",type=["csv"], key = 'uploader')
            
            if file is not None:
                st.session_state['file_path'] = 'csv'
                file_details = {"nome do arquivo":st.session_state['file_path'],"tipo do arquivo":file.type}
                
                try:
                    st.session_state['df'] = load_csv_data(file)
                except:
                    st.warning('Erro no upload... Tente novamente!')
                    st.stop()
                    
                st.session_state['id'] = st.selectbox("Identificador:", st.session_state['df'].columns)
                st.session_state['time_col'] = st.selectbox("Coluna Temporal:", st.session_state['df'].columns)
                st.session_state['real'] = st.selectbox("Real :", st.session_state['df'].columns)
                st.session_state['previsto'] = st.selectbox("Previsto :", st.session_state['df'].columns)
                st.session_state['previsto_compare'] = st.multiselect("Previsto (Benchmark):", st.session_state['df'].columns)
                st.session_state['classe'] = st.selectbox("Classe:", st.session_state['df'].columns)
                st.session_state['agrupamento'] = st.selectbox("Agrupamento:",['NÃO']+st.session_state['df'].columns.tolist())
                st.session_state['df']['NÃO'] = 0
                
                file_path = st.session_state['file_path']
                data_group = st.session_state['id']
                time_col = st.session_state['time_col']
                y_true  = st.session_state['real']
                y_predicted = st.session_state['previsto']
                y_benchmark = st.session_state['previsto_compare']
                classe = st.session_state['classe']
                data_group2 = st.session_state['agrupamento']
                grouped_df = st.session_state['df'][[data_group,data_group2,time_col,y_true,y_predicted]]
                
                st.session_state['chosen_group'] = st.selectbox(f"Selecione o agrupamento:",
                                sorted(st.session_state['df'][data_group2].unique().tolist()))
                chosen_group = st.session_state['chosen_group']
                
                df = st.session_state['df'][st.session_state['df'][data_group2]==chosen_group]  
                df = preprocess_dataframe(df,time_col,y_true,y_predicted,y_benchmark)
                    
            else:
                st.warning('Selecione o arquivo e preencha os campos')
                st.stop()
    
    return df,file_path,grouped_df,time_col,data_group,data_group2,chosen_group,classe,y_true,y_predicted,y_benchmark

def filter_by_period(dataframe:pd.DataFrame,
                     time_col:str):
    
    try:
        st.subheader('Intervalo:')
        start_date, end_date = st.slider('',
                            value=[dataframe[time_col].min(), dataframe[time_col].max()],
                            max_value = dataframe[time_col].max(),
                            min_value = dataframe[time_col].min(),
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
        period_mask = (dataframe[time_col] >= start_date) & (dataframe[time_col] <= end_date)
        updated_df = dataframe.loc[period_mask].copy()
    
    except:
        updated_df = dataframe.copy()
        st.warning('Falha na data')
        
        st.stop()
        
    finally:
        return updated_df
        