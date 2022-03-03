import numpy as np
import pandas as pd

from dashboard import *
from utils import *
from pages.page1 import create_page1
from pages.page2 import create_page2
from pages.page3 import create_page3

from pages import MENU

# MENU ALTERNATIVO DE TESTE
def main():
    choice = st.sidebar.radio("", MENU)
    df, time_col, data_group, y_true, y_predicted = load_csv_file()
    df = create_initial_page(df, time_col, y_true, y_predicted)

    if choice == MENU[0]:
        create_page1(df, data_group, y_true, y_predicted)
    
    elif choice == MENU[1]:
        create_page2(df, data_group, time_col, y_true, y_predicted) 

if __name__ == "__main__":
    set_streamlit()
    main()
