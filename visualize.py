
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')

import dash
import plotly.graph_objects as go
import dash_html_components as html
import dash_core_components as dcc
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

from collections import deque
from multiprocessing import Process, Queue

import networkx as nx

from neat.neatTypes import NeuronType

node_values = []
images = []
last_data = []

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# app.config['suppress_callback_exceptions'] = True
dash_queue = Queue()

app.layout = html.Div([
    dcc.Store(id='storage', storage_type='memory'),
    # dcc.Interval(id='my-interval', interval=1 * 1000),
    # html.Div(id='my-output-interval'),
    html.Button('Refresh', id='refresh-button'),
    dcc.Dropdown(id='dropdown'),
    html.Div(children=[
            dcc.Graph(id='network', figure=go.Figure()),
        ],
        style={'display': 'inline-block', 'width': '60%'}
    )

    # dcc.Slider(id='time_range', min=0, value=0),

])

# @app.callback(Output('storage', 'data'), [Input('my-interval', 'n_intervals')], [State('storage', 'data')])
@app.callback(Output('storage', 'data'), [Input('refresh-button', 'n_clicks')], [State('storage', 'data')])
def on_interval(n, data):
    if dash_queue.empty():
        # prevent the None callbacks is important with the store component.
        # you don't want to update the store for nothing.
        raise PreventUpdate

    data = data or {'network': [], 'activations': []}

    while not dash_queue.empty():
        queue_data = dash_queue.get()
        G = queue_data[0]

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

        data['network'].append({
            'n': len(queue_data[1]),
            'node_x': node_x, 'node_y': node_y, 'node_type': [n[1]['type'].value for n in G.nodes.data()],
            'edge_x': edge_x, 'edge_y': edge_y,
            'weights': [e[2]['weight'] for e in G.edges().data()]
        })
        data['activations'].append(queue_data[1])

    print("Updating session storage: {}".format(len(data['network'])))

    return data


@app.callback(Output('dropdown', 'options'), [Input('storage', 'modified_timestamp')], [State('storage', 'data')])
def update_dropdown(ts, data):
    if ts is None:
        raise PreventUpdate

    print("Updating dropdown. Runs: {}".format(len(data)))
    return [{'label': "Run #{}".format(i), 'value': i} for i in range(len(data['network']))]

@app.callback(Output('network', 'figure'),
              [Input('dropdown', 'value')],
              [State('storage', 'data'), State('network', 'figure')])
def update_graph_live(dropdown_i, data, fig):
    if None in [data, dropdown_i]:
        raise PreventUpdate

    print("Drawing network #{}".format(dropdown_i))

    network = data['network'][dropdown_i]
    activations = data['activations'][dropdown_i]

    n_steps = len(activations)
    n_nodes = len(network['node_x'])
    weights = network['weights']

    edge_trace = go.Scatter(
        x=network['edge_x'], y=network['edge_y'],
        line=dict(width=1.5, color='#888'),
        hoverinfo='none',
        mode='lines',
    )

    node_size = 20

    all_node_traces = []
    all_annotations = []
    for i in range(n_steps):
        scatter = go.Scatter(
            x=network['node_x'],
            y=network['node_y'],
            mode="markers",
            # text=np.around(activations[i], 2),
            # textposition="middle right",
            # hoverinfo='text',
            marker=dict(
                showscale=True,
                reversescale=True,
                colorscale='rdbu',
                color=activations[i],
                cmid=0,
                size=node_size,
                # size=np.array(node_sizes[i])*10 + 10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=2)
            )
        )

        annotations = []
        for j, (x, y) in enumerate(zip(network['node_x'], network['node_y'])):

            location = 'middle'
            node_type = NeuronType(network['node_type'][j])
            # print(node_type, NeuronType.INPUT, NeuronType(node_type) == NeuronType.INPUT)
            # if node_type == NeuronType.INPUT:
            #     location = 'bottom'
            # elif node_type == NeuronType.OUTPUT:
            #     location = 'top'

            annotation = go.layout.Annotation(
                    x=x,
                    y=y,
                    text=str(np.round(activations[i][j], 2)),
                    showarrow=False,
                    # xanchor='center',
                    yref='y',
                    yshift=-(node_size) if node_type == NeuronType.INPUT else (node_size),
                    yanchor=location
                )

            annotations.append(annotation)

        all_node_traces.append(scatter)
        all_annotations.append(annotations)

    steps = []
    for i in range(len(activations)):
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
            ],
        )
        steps.append(step)

    sliders = [go.layout.Slider(
        active=0,
        currentvalue={"prefix": "Step: "},
        # pad={"t": 50},
        steps=steps
    )]

    start_data = [all_node_traces[0]]
    start_data.extend(all_annotations[0])
    start_data.append(edge_trace)
    fig = go.Figure(
        data=[all_node_traces[0], edge_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            sliders=sliders,
            # annotations=[dict(
            #     showarrow=False,
            #     xref="paper", yref="paper",
            #     x=0.005, y=-0.002
            # )],
            annotations=all_annotations[0],
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
        frames=[go.Frame(data=[n], name=str(i), layout=go.Layout(annotations=a)) for i, (n, a) in enumerate(zip(all_node_traces, all_annotations))]
    )

    w_i = 0
    for i in range(0, len(network['edge_x']) - 1, 3):
        x0 = network['edge_x'][i]
        x1 = network['edge_x'][i + 1]
        y0 = network['edge_y'][i]
        y1 = network['edge_y'][i + 1]

        # print(x0, x1, y0, y1)

        fig.add_annotation(
            go.layout.Annotation(
                x=(x0+x1)/2,
                y=(y0+y1)/2,
                text=str(np.round(weights[w_i], 2)),
                # showarrow=False,
                # xanchor='left',
                # yanchor='bottom',
                # xshift=5,
                # yshift=5
            )
        )

        w_i += 1

    return fig

class Visualize(Process):

    def __init__(self):
        super(Visualize, self).__init__()

    def run(self):
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

        app.run_server(debug=False, processes=1, threaded=False)