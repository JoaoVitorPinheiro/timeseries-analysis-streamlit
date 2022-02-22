import streamlit as st
from analysis import *
import os
   
PLOT_HEIGHT = 500
PLOT_WIDTH = 650
BACKGROUND_COLOR = 'white'
COLOR = 'black'

def set_streamlit():
    st.set_page_config(
    page_title = "Análise do Forecast",
    page_icon = "📈",
    layout="wide",)
    return "initializing..."

# NÃO FUNFA AINDA
def set_page_container_style(
        max_width: int = 1100, max_width_100_percent: bool = False,
        padding_top: int = 1, padding_right: int = 10, padding_left: int = 1, padding_bottom: int = 10,
        color: str = COLOR, background_color: str = BACKGROUND_COLOR,
    ):
        if max_width_100_percent:
            max_width_str = f'max-width: 100%;'
        else:
            max_width_str = f'max-width: {max_width}px;'
        st.markdown(
            f'''
            <style>
                .reportview-container .css-1lcbmhc .css-1outpf7 {{
                    padding-top: 35px;
                }}
                .reportview-container .main .block-container {{
                    {max_width_str}
                    padding-top: {padding_top}rem;
                    padding-right: {padding_right}rem;
                    padding-left: {padding_left}rem;
                    padding-bottom: {padding_bottom}rem;
                }}
                .reportview-container .main {{
                    color: {color};
                    background-color: {background_color};
                }}
            </style>
            ''',
            unsafe_allow_html=True,
        )

def format_fig(target_fig,
               title_text = '',
               x_title = '',
               y_title = '',
               h = PLOT_HEIGHT,
               w = PLOT_WIDTH):
    
        target_fig.update_layout(
                yaxis_title=y_title,
                xaxis_title=x_title,
                legend_title_text="",
                height=h,
                width=w,
                #title_text=title_text,
                title_x=0.5,
                title_y=1,
                hovermode="x",
                hoverlabel=dict(
                bgcolor="white",
                font_size=16,
                font_family="Rockwell"
                #paper_bgcolor='rgba(0,0,0,0)',
                #plot_bgcolor='rgba(0,0,0,0)
                )
    )
        return target_fig
    
def open_streamlit():
    # tests
    os.system('streamlit run app.py')
    return 
