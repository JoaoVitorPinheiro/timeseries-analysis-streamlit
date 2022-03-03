from dashboard import *
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from typing import List, Any, Dict, Tuple
from kpi import MAPE, RMSE

def create_page1(df, data_group, y_true, y_predicted):
    pass  
    
def create_global_metrics(data:pd.DataFrame, time_col:str, data_group:str, classes:List, y_true:str, y_predicted:str):
    
    #st.markdown(f"""
    #        <marquee style='width: 100%; color: rgb(234, 82, 111);'><b> <font size="5">{categories}.</b></font></marquee>""",
    #unsafe_allow_html = True)
    
    st.markdown("""
                <span style="color:rgb(234, 82, 111)"><font size="5">DIAS ACIMA DE 5%</font></span>""",
        unsafe_allow_html = True)
    
    #DIAS ACIMA DE 5%
    with st.expander("..."):
        st.subheader(data_group)
        dfplot = data.groupby([data_group]).apply(lambda x: 100*x.acima5.sum()/x.acima5.count()).reset_index()
        
        fig = go.Figure(data=[go.Bar(x=dfplot[data_group].unique().tolist(),
                                    y=dfplot.iloc[:, 1], text = dfplot.iloc[:,[1]])])
        # Customize aspect
        fig.update_xaxes(tickangle=-45)
        fig.update_traces(marker_color='rgb(234, 82, 111)', marker_line_color='rgb(0, 0, 0)',
                    marker_line_width=1.5, opacity=0.75, texttemplate='%{text:.1f}', textposition='outside')
        fig.update_layout(hovermode='x')          
        fig = format_fig(fig, x_title=data_group, y_title='Percentual(%)')
        st.plotly_chart(fig, use_container_width=True)
        
        # ACIMA DE 5% POR CLASSE
        for classe in classes:
            st.subheader(classe)
            dfplot = data.groupby([time_col, classe]).sum().reset_index()
            dfplot['mape'] = MAPE(dfplot[y_true], dfplot[y_predicted])
            dfplot['acima5'] = np.where(dfplot['mape']>5, True, False)
            dfplot = dfplot.groupby([classe]).apply(lambda x: 100*x.acima5.sum()/x.acima5.count()).reset_index()
                        
            fig = go.Figure(data=[go.Bar(x=dfplot[classe].unique().tolist(),
                                        y=dfplot.iloc[:, 1])])
            # Customize aspect
            fig.update_xaxes(tickangle=-45)
            fig.update_traces(marker_color='rgb(234, 82, 111)', marker_line_color='rgb(0, 0, 0)',
                        marker_line_width=1.5, opacity=0.75,
                        texttemplate='%{y:.1f}', textposition='outside')
            fig.update_layout(hovermode='x')          
            fig = format_fig(fig, x_title=data_group, y_title='Percentual(%)')
            st.plotly_chart(fig, use_container_width=True)
            
    st.markdown("""
                <span style="color:rgb(110, 68, 255)"><font size="5">MAPE</font></span>""",
        unsafe_allow_html = True)
    #MAPE
    with st.expander("..."):
        
        st.subheader(data_group)
        dfplot = data
        dfplot["mape"] = np.where(dfplot["mape"]>100, 100, dfplot["mape"])
        dfplot = dfplot.groupby([data_group]).mean().reset_index()
        #dfplot["mape"] = np.where(dfplot["mape"]>100, np.nan, dfplot["mape"])
        fig = go.Figure(data=[go.Bar(x=dfplot[data_group].unique().tolist(),
                                    y=dfplot["mape"])])
        # Customize aspect
        fig.update_xaxes(tickangle=-45)
        fig.update_traces(marker_color='rgb(110, 68, 255)', marker_line_color='rgb(0, 0, 0)',
                    marker_line_width=1.5, opacity=0.75,
                    texttemplate='%{y:.1f}', textposition='outside')
        fig.update_layout(hovermode='x')          
        fig = format_fig(fig, x_title=data_group, y_title='Percentual(%)')
        st.plotly_chart(fig, use_container_width=True)
        
        # MAPE POR CLASSE
        for classe in classes:
            
            st.subheader(classe)
            dfplot = data.groupby([time_col, classe]).sum().reset_index()
            dfplot['mape'] = MAPE(dfplot[y_true], dfplot[y_predicted])
            dfplot["mape"] = np.where(dfplot["mape"]>100, 100, dfplot["mape"])
            dfplot = dfplot.groupby([classe]).mean().reset_index()
                  
            fig = go.Figure(data=[go.Bar(x=dfplot[classe].unique().tolist(),
                                        y=dfplot['mape'])])

            fig.update_xaxes(tickangle=-45)
            fig.update_traces(marker_color='rgb(110, 68, 255)', marker_line_color='rgb(0, 0, 0)',
                        marker_line_width=1.5, opacity=0.75,
                        texttemplate='%{y:.1f}', textposition='outside')
            fig.update_layout(hovermode='x')          
            fig = format_fig(fig, x_title=data_group, y_title='Percentual(%)')
            st.plotly_chart(fig, use_container_width=True)
            
    st.markdown("""
            <span style="color:rgb(37, 206, 209)"><font size="5">RMSE</font></span>""",
    unsafe_allow_html = True)
    
    #RMSE
    with st.expander("..."):
        
        dfplot = data.groupby([data_group]).apply(lambda x: RMSE(x[y_true], x[y_predicted])).reset_index()
        fig = go.Figure(data=[go.Bar(x=dfplot[data_group].unique().tolist(),
                                    y=dfplot.iloc[:, 1], text = dfplot.iloc[:,[1]])])

        fig.update_xaxes(tickangle=-45)
        fig.update_traces(marker_color='rgb(37, 206, 209)', marker_line_color='rgb(0, 0, 0)',
                    marker_line_width=1.5, opacity=0.75, texttemplate='%{text:.0f}', textposition='outside')
        fig.update_layout(hovermode='x')          
        fig = format_fig(fig, x_title=data_group, y_title='rmse')
        st.plotly_chart(fig, use_container_width=True)
