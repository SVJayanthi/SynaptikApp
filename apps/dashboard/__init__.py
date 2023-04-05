"""Instantiate a Dash app."""
import dash
from dash import dash_table
from dash import dcc
from dash.dependencies import Output, Input

from dash import html
from plotly.graph_objs import *

from .data import create_dataframe

import plotly
import random
import plotly.graph_objs as go
import time
import pandas as pd
from networkx.readwrite import json_graph
import plotly.graph_objects as go

import plotly.express as px
from plotly.subplots import make_subplots

from PIL import Image
import numpy as np
from multiprocessing import Process
from threading import Thread
import json

from apps.dashboard import *

APP_ID = 'dash_app_1'
URL_BASE = '/dashapp/'
MIN_HEIGHT = 200

X = []
X.append(1)
  
Y = []
Y.append(1)

class DashApp():
    def __init__(self, global_dict):
        self.global_dict = global_dict

    def init_dash(self, server):
        """Create a Plotly Dash dashboard."""
        self.dash_app = dash.Dash(
            server=server,
            routes_pathname_prefix="/dashapp/",
            external_stylesheets=[
                "/static/dist/css/styles.css",
                "https://fonts.googleapis.com/css?family=Lato",
            ],
        )

        # Load DataFrame
        df = create_dataframe()

        # Custom HTML layout
        # dash_app.index_string = html_layout

        # Create Layout
        self.dash_app.layout = html.Div(
            children=[
                dcc.Graph(id = 'live-graph', animate = False),
                dcc.Interval(
                    id = 'graph-update',
                    interval = 1000,
                    n_intervals = 0
                ),
            ],
            id="dash-container",
        )

        self.init_callbacks(self.dash_app)
        return self.dash_app

    def init_dashboard(self, server):
        self.init_dash(server)
        return self.dash_app.server

    def init_callbacks(self, dash_app):
        @dash_app.callback(
        # Callback input/output
            Output('live-graph', 'figure'),
            [ Input('graph-update', 'n_intervals') ]
        )
        def update_graph_scatter(n):
            graphJSON = self.global_dict.get('graphJSON')

            graph = json.loads(graphJSON)
            # print(graph)
            layout =go.Layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                font = dict(color = '#00FFFF'),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))

            fig = Figure(data=graph['data'], layout=layout)
            return fig
        

class DashApp1():
    def __init__(self, global_dict):
        self.global_dict = global_dict

    def init_dash(self, server):
        """Create a Plotly Dash dashboard."""
        self.dash_app = dash.Dash(
            server=server,
            routes_pathname_prefix="/dashapp1/",
            external_stylesheets=[
                "/static/dist/css/styles.css",
                "https://fonts.googleapis.com/css?family=Lato",
            ],
        )

        # Load DataFrame
        df = create_dataframe()

        # Custom HTML layout
        # dash_app.index_string = html_layout

        # Create Layout
        self.dash_app.layout = html.Div(
            children=[
                dcc.Graph(id = 'live-graph', animate = False),
                dcc.Interval(
                    id = 'graph-update',
                    interval = 1000,
                    n_intervals = 0
                ),
            ],
            id="dash-container",
        )

        self.init_callbacks(self.dash_app)
        return self.dash_app

    def init_dashboard(self, server):
        self.init_dash(server)
        return self.dash_app.server

    def init_callbacks(self, dash_app):
        @dash_app.callback(
        # Callback input/output
            Output('live-graph', 'figure'),
            [ Input('graph-update', 'n_intervals') ]
        )
        def update_graph_scatter(n):
            costJSON = self.global_dict.get('costJSON')

            graph = json.loads(costJSON)

            return {"data": graph['data'], "layout": graph['layout']}
            # fig = Figure(data=graph['data'], layout=graph['layout'])
            # return fig
        

class DashApp2():
    def __init__(self, global_dict):
        self.global_dict = global_dict

    def init_dash(self, server):
        """Create a Plotly Dash dashboard."""
        self.dash_app = dash.Dash(
            server=server,
            routes_pathname_prefix="/dashapp2/",
            external_stylesheets=[
                "/static/dist/css/styles.css",
                "https://fonts.googleapis.com/css?family=Lato",
            ],
        )

        # Load DataFrame
        df = create_dataframe()

        # Custom HTML layout
        # dash_app.index_string = html_layout

        # Create Layout
        self.dash_app.layout = html.Div(
            children=[
                dcc.Graph(id = 'live-graph', animate = False),
                dcc.Interval(
                    id = 'graph-update',
                    interval = 1000,
                    n_intervals = 0
                ),
            ],
            id="dash-container",
        )

        self.init_callbacks(self.dash_app)
        return self.dash_app

    def init_dashboard(self, server):
        self.init_dash(server)
        return self.dash_app.server

    def init_callbacks(self, dash_app):
        @dash_app.callback(
        # Callback input/output
            Output('live-graph', 'figure'),
            [ Input('graph-update', 'n_intervals') ]
        )
        def update_graph_scatter(n):
            metricsJSON = self.global_dict.get('metricsJSON')

            graph = json.loads(metricsJSON)
            return {"data": graph['data'], "layout": graph['layout']}
            # fig = Figure(data=graph['data'], layout=graph['layout'])
            # return fig
       