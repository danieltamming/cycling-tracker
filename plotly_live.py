import random
import time
from itertools import count
from multiprocessing import Process, Queue

import pandas as pd
import numpy as np
from plotly.offline import plot
import plotly.graph_objs as go

x=np.array([2,5,8,0,2,-8,4,3,1])
y=np.array([2,5,8,0,2,-8,4,3,1])


data = [go.Scatter(x=x,y=y)]
fig = go.Figure(data = data,layout = go.Layout(title='Offline Plotly Testing',width = 800,height = 500,
                                           xaxis = dict(title = 'X-axis'), yaxis = dict(title = 'Y-axis')))


# plot(fig,show_link = False)

import plotly.graph_objects as go # or plotly.express as px
# fig = go.Figure() # or any Plotly Express function e.g. px.bar(...)
# fig.add_trace( ... )
# fig.update_layout( ... )

import dash
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash()
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

app.run_server(debug=True, use_reloader=True)  # Turn off reloader if inside Jupyter