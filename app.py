from dashboard import *
from utils import *

from pages import page1,page2,page3,page4

os.environ['TZ'] = 'UTC'

MENU = ['Métricas Globais',
        #'Agrupamentos',
        'Análise de Resíduos',
        'Benchmark']
     
def main():
    
    set_streamlit()
    set_page_style()
    
    session_state_vars = ['file_path','id',
                          'time_col','real',
                          'previsto','previsto_compare',
                          'classe','agrupamento',
                          'chosen_group','selected',
                          'chosen_col','chosen_item',
                          'df','navigator',
                          'password']
    
    for variavel in session_state_vars:
        if variavel not in st.session_state:
            st.session_state[variavel] = None
              
    st.sidebar.title("Páginas")
    st.session_state['navigator'] = st.sidebar.radio("", MENU)
    choice = st.session_state['navigator']
    
    df,file_path,grouped_df,time_col,data_group,data_group2,chosen_group,classe,y_true,y_predicted,y_benchmark = init_file_upload()
    
    if file_path:
        
        updated_df = filter_by_period(df,time_col)
        
        with st.expander("Dados"):
            st.dataframe(updated_df[[data_group,data_group2,time_col,y_true,y_predicted]+[classe]])

        if choice == 'Métricas Globais':
            page1.open_page(updated_df,time_col,data_group,chosen_group,classe,y_true,y_predicted)

        elif choice == 'Agrupamentos':  #Not active
            page2.open_page(grouped_df,time_col,data_group,data_group2,y_true,y_predicted)

        elif choice == 'Análise de Resíduos': 
            page3.open_page(updated_df,time_col,data_group,classe,y_true,y_predicted)

        elif choice == 'Benchmark':
            page4.open_page(updated_df,time_col,data_group,classe,y_true,y_benchmark)
        
if __name__ == "__main__":
    main()
