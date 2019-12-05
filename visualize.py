import networkx as nx

from queue import Queue

import sys

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import style

import plotly.express as px
from plotly.offline import plot
import dash
import plotly.graph_objects as go
import dash_html_components as html
import dash_core_components as dcc
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

from collections import deque
from multiprocessing import Process, Queue

import datetime

from neat.phenotypes import Phenotype
from neat.utils import Singleton
# from graph_tool.all import *

node_values = []
images = []
last_data = []

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# app.config['suppress_callback_exceptions'] = True
dash_queue = Queue()

app.layout = html.Div([
    dcc.Store(id='storage', storage_type='memory'),
    dcc.Interval(id='my-interval', interval=1 * 1000),
    html.Div(id='my-output-interval'),
    dcc.Dropdown(id='dropdown'),
    dcc.Graph(id='network', figure=go.Figure()),
    # dcc.Slider(id='time_range', min=0, value=0),

])

@app.callback(Output('storage', 'data'), [Input('my-interval', 'n_intervals')], [State('storage', 'data')])
def on_interval(n, data):
    if dash_queue.empty():
        # prevent the None callbacks is important with the store component.
        # you don't want to update the store for nothing.
        raise PreventUpdate

    # print(data)
    data = data or {'network': [], 'node_sizes': []}
    # print(data)
    # print(len(data['images']))
    queue_data = dash_queue.get()
    # print(queue_data[1])
    # images = data['images']
    # images.append(queue_data[2][1][1][1])
    # data['node_sizes'].append(queue_data[1])
    G = queue_data[0]
    # node_adjacencies = []
    # for node, adjacencies in enumerate(G.adjacency()):
    #     node_adjacencies.append(len(adjacencies[1]))

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    data['network'].append({'n': len(queue_data[1]), 'node_x': node_x, 'node_y': node_y, 'edge_x': edge_x, 'edge_y': edge_y})
    data['node_sizes'].append(queue_data[1])

    print("Updating session storage: {}".format(len(data['network'])))

    return data


@app.callback(Output('dropdown', 'options'), [Input('storage', 'modified_timestamp')], [State('storage', 'data')])
def update_dropdown(ts, data):
    if ts is None:
        raise PreventUpdate

    print("Updating dropdown. Runs: {}".format(len(data)))
    return [{'label': "Run #{}".format(i), 'value': i} for i in range(len(data['network']))]

@app.callback(Output('my-output-interval', 'children'), [Input('my-interval', 'n_intervals')])
def display_output(n):
    now = datetime.datetime.now()
    return '{} intervals have passed. It is {}:{}:{}'.format(
        n,
        now.hour,
        now.minute,
        now.second
    )

# @app.callback([Output('time_range', 'max'), Output('time_range', 'marks')], [Input('dropdown', 'value')], [State('storage', 'data')])
# def update_graph_live(i, data):
#     if data is None or i is None:
#         raise PreventUpdate
#
#     network = data['network'][i]
#     n = network['n']
#
#     return n, {j: j for j in range(n)}


@app.callback(Output('network', 'figure'),
              # [Input('dropdown', 'value'), Input('time_range', 'value')],
              [Input('dropdown', 'value')],
              [State('storage', 'data'), State('network', 'figure')])
def update_graph_live(dropdown_i, data, fig):
    if None in [data, dropdown_i]:
        raise PreventUpdate

    print("Drawing network #{}".format(dropdown_i))

    network = data['network'][dropdown_i]
    # print(slider_i, len(data['node_sizes'][dropdown_i]))
    node_sizes = data['node_sizes'][dropdown_i]
    n_nodes = network['n']
    graph_steps = len(data['node_sizes'][dropdown_i])

    edge_trace = go.Scatter(
        x=network['edge_x'], y=network['edge_y'],
        line=dict(width=1.5, color='#888'),
        hoverinfo='none',
        mode='lines')


    all_node_traces = []
    steps = []
    for i in range(n_nodes):
        scatter = go.Scatter(
            x=network['node_x'],
            y=network['node_y'],
            mode='markers',
            # hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                reversescale=True,
                color=[],
                size=np.array(node_sizes[i])*15 + 5,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=2)
            )
        )
        all_node_traces.append(scatter)

        step = dict(
            # Update method allows us to update both trace and layout properties
            method='animate',
            label=i,
            args=[
                [i],
                # Make the ith trace visible
                {
                    "frame": {"duration": 300, "redraw": False},
                    "mode": "immediate", "transition": {"duration": 300}
                },

                # Set the title for the ith trace
                # {'title.text': str(i)}
            ],
        )
        steps.append(step)


    sliders = [go.layout.Slider(
        active=0,
        currentvalue={"prefix": "Step: "},
        # pad={"t": 50},
        steps=steps
    )]

    return go.Figure(
        data=[all_node_traces[0], edge_trace],
        layout=go.Layout(
            # title='<br>Network Graph of '+str(len(data[0]))+' neurons',
            # titlefont=dict(size=16),
            # showlegend=False,
            hovermode='closest',
            # margin=dict(b=20,l=5,r=5,t=40),
            sliders=sliders,
            annotations=[dict(
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002
            )],
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play",
                              method="animate",
                              args= [None, {"frame": {"duration": 500, "redraw": False},
                                            "fromcurrent": True,
                                            "transition": {"duration": 300, "easing": "quadratic-in-out"}}
                                     ],
                        )]
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        ),
        frames=[go.Frame(data=[n]) for n in all_node_traces]
    )

class Visualize(Process):

    def __init__(self):
        super(Visualize, self).__init__()
        # global queue
        # queue = main_queue

        self.data = []

        self.video_size = (0, 0)


    def run(self):
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

        app.run_server(debug=False, processes=4, threaded=False)