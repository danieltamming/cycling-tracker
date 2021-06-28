import datetime
import webbrowser
from threading import Timer
import base64
import pickle

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly
# import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import pandas as pd

import cv2

# pip install pyorbital
# from pyorbital.orbital import Orbital
# satellite = Orbital('TERRA')

df = pd.read_csv('data.csv')
cap = cv2.VideoCapture('../cycling-tracker-private/videos/short_vid3.MOV')
# with open('data.pickle', 'rb') as f:
#     q = pickle.load(f)

frame_list = []
for _ in range(10*30):
    ret, frame = cap.read()
    if not ret:
        break
    frame_list.append(frame)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    html.Div([
        html.H4('Cycling Tracker'),
        html.Div(id='live-update-text'),
        html.Div(id='image'),
        dcc.Graph(
            id='speed-graph',
            # figure=dict(layout=dict(height=700))
        ),
        dcc.Interval(
            id='interval-component',
            interval=1/10*1000, # in milliseconds
            n_intervals=0
        ),
        dcc.Store(id='intermediate-value')
    ])
)

@app.callback(
    Output('intermediate-value', 'data'),
    Input('interval-component', 'n_intervals'))
def clean_data(n):
    return n


@app.callback(Output('live-update-text', 'children'),
              Input('intermediate-value', 'data'))
def update_metrics(n):
    print(n)
    # lon, lat, alt = satellite.get_lonlatalt(datetime.datetime.now())
    frame_count, second, rpm, speed, distance = df.iloc[n]
    style = {'padding': '5px', 'fontSize': '16px'}
    return [
        html.Span('Frame: {0:.2f}'.format(frame_count), style=style),
        html.Span('Second: {0:.2f}'.format(second), style=style),
        html.Span('RPM: {0:0.2f}'.format(rpm), style=style)
    ]


# Multiple components can update everytime interval gets fired.
@app.callback(Output('speed-graph', 'figure'),
              Input('intermediate-value', 'data'))
def update_graph_live(n):
    n = int(n / 10)
    fig = plotly.tools.make_subplots(rows=1, cols=1, vertical_spacing=0.2)
    y_min = min(20, df['speed'].iloc[:n+1].min())
    y_max = max(30, df['speed'].iloc[:n+1].max())
    fig.update_layout(yaxis_range=[y_min, y_max])
    fig.append_trace({
        'x': df['second'].iloc[:n+1],
        'y': df['speed'].iloc[:n+1],
        'name': 'Altitude',
        'mode': 'lines+markers',
        'type': 'scatter'
    }, row=1, col=1)
    return fig

# Multiple components can update everytime interval gets fired.
@app.callback(Output('image', 'children'),
              Input('intermediate-value', 'data'))
def update_graph_live(n):
    # for _ in range(1):
    #     ret, frame = cap.read()
    # ret, frame = cap.read()
    # if not ret:
    #     return dash.no_update
    frame = frame_list[3*n]
    k = 0.4
    new_width = int(k * frame.shape[1])
    new_height = int(k * frame.shape[0])
    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    retval, my_buffer = cv2.imencode('.jpg', frame)
    encoded_image = base64.b64encode(my_buffer)
    return [html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()))]

def open_webpage():
    webbrowser.open_new('http://localhost:8050/')

if __name__ == '__main__':
    Timer(3, open_webpage).start()
    # open_webpage()
    app.run_server(debug=False, use_reloader=False)