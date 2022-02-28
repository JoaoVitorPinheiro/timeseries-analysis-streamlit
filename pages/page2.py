import plotly.graph_objects as go
from utils import preprocess_dataframe
from typing import List, Any, Dict, Tuple
from dashboard import *

def create_page2(df, data_group, time_col, y_true, y_predicted):
    pass

def create_grouped_radar(data, data_group, data_group2, time_col, y_true:str, y_predicted:str):
    categories = sorted(data[data_group].unique().tolist())[:-1]
    groups = sorted(data[data_group2].unique().tolist())

    options = st.multiselect(
            'Adicione os Itens:',
            options = categories,
            default = categories)

    chosen_metric=st.selectbox('Métrica', ['MAPE', 'ACIMA5'])
    #st.markdown("""
    #        <span style="color:rgb(32,4,114)"> <font size="5">MAPE</font></span>""",
    #unsafe_allow_html = True)
    fig = go.Figure()
    for group in groups:
        
        dfplot = data[data[data_group2]==group]
        dfplot = preprocess_dataframe(dfplot, time_col, y_true, y_predicted)
        dfplot['mape'] = dfplot['mape'].clip(0,100)
        if chosen_metric=='MAPE':
            values = dfplot.groupby([data_group]).mape.mean()
        else:
            values = dfplot.groupby([data_group]).apply(lambda x: 100*x.acima5.sum()/x.acima5.count())
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=sorted(options),
            fill='toself',
            opacity=0.65,
            mode = 'lines+markers',
            name=group
        ))
    fig.update_layout(
        template="plotly_white",
        polar=dict(
            radialaxis=dict(
            visible=True,
        )),
    showlegend=True,
    height=750, width=750,
    )
    st.plotly_chart(fig, use_container_width=True)
