import datetime
import webbrowser
from threading import Timer
import base64

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

ret, frame = cap.read()
retval, my_buffer = cv2.imencode('.jpg', frame)
encoded_image = base64.b64encode(my_buffer)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    html.Div([
        html.H4('TERRA Satellite Live Feed'),
        html.Div(id='live-update-text'),
        dcc.Graph(
            id='speed-graph',
            # figure=dict(layout=dict(height=700))
        ),
        # dcc.Graph(
        #     id="image-graph", 
        #     figure = go.Figure(go.Image(z=frame)), 
        #     style={"width": "75%", "display": "inline-block"}
        # )
        html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode())),
        dcc.Interval(
            id='interval-component',
            interval=5*1000, # in milliseconds
            n_intervals=0
        )
    ])
)


@app.callback(Output('live-update-text', 'children'),
              Input('interval-component', 'n_intervals'))
def update_metrics(n):
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
              Input('interval-component', 'n_intervals'))
def update_graph_live(n):
    fig = plotly.tools.make_subplots(rows=1, cols=1, vertical_spacing=0.2)
    y_min = min(20, df['speed'].iloc[:n+1].min())
    y_max = max(30, df['speed'].iloc[:n+1].max())
    fig.append_trace({
        'x': df['second'].iloc[:n+1],
        'y': df['speed'].iloc[:n+1],
        'name': 'Altitude',
        'mode': 'lines+markers',
        'type': 'scatter'
    }, row=1, col=1)
    return fig

# Multiple components can update everytime interval gets fired.
# @app.callback(Output('image-graph', 'figure'),
#               Input('interval-component', 'n_intervals'))
# def update_graph_live(n):
#     # fig = plotly.tools.make_subplots(rows=2, cols=1, vertical_spacing=0.2)
#     ret, frame = cap.read()
#     fig = px.imshow(frame[:, :, ::-1])
#     return fig

def open_webpage():
    webbrowser.open_new('http://localhost:8050/')

if __name__ == '__main__':
    Timer(3, open_webpage).start()
    # open_webpage()
    app.run_server(debug=True, use_reloader=False)