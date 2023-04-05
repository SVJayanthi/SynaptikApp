

from apps.home import blueprint
from flask import render_template, request, current_app
from flask import url_for, session, jsonify

from flask_login import login_required
from jinja2 import TemplateNotFound
from flask import current_app
from flask import request
import json
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import plotly
import os
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
def plotly_data_vis(data):
    names = ['Standard', 
            'Devices without Storage', 
            'Devices without Storage & PV']
    
    time_of_day = 0.01*np.random.rand(len(data[0]))
    cumulative_cost = (np.clip(data[0], 0, None))*time_of_day
    df = pd.DataFrame({'Energy Expenditures': cumulative_cost, 'Time of Day Rate': time_of_day})
    # df = pd.DataFrame([[], [time_of_day]], columns=["Cumulative Expenditures", "Time of Day Rate"])
    
    fig = px.line(df)

    fig.update_layout(
        title="<b>Factory Wide Energy Expenditures</b>",
        xaxis_title="<b>Time (hours)</b>",
        yaxis_title="<b>Cost ($)</b>",
        height=450,
        margin=dict(l=80, r=20, t=50, b=60),
        # yaxis_title="Y Axis Title",
        legend_title="Cost",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            xanchor="right",
            y=-0.3,
            x=1
        ),
        font=dict(
            size=12,
            color="White"
        ), 
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    fig.update_yaxes(automargin=True)

    costJSON1 = json.dumps(fig, cls = plotly.utils.PlotlyJSONEncoder)
    

    # sub_fig3 = make_subplots(rows=2, cols=2, shared_xaxes=True, vertical_spacing=0.02)
    
    # df = pd.DataFrame(np.swapaxes(data, 0, 1), columns=names)
    df = pd.DataFrame(np.clip(np.swapaxes(data, 0, 1), 0, None), columns=names)

    # df.insert(0, "names", names)
    fig = px.area(df)
    fig.for_each_annotation(lambda a: a.update(text='kWh'))

    # fig.for_each_annotation(lambda a: '')# a.update(text=a.text.split("=")[-1]))

    # fig['layout']['yaxis']['title']= dict(text="GDP-per-capita", font=dict(size=5))

    # fig['layout']['yaxis']['title']['text']=names[2]
    # fig['layout']['yaxis2']['title']['text']=names[1]
    # fig['layout']['yaxis3']['title']['text']=names[0]

    
    # fig = go.Figure(data = fig2.data + fig2.data)

    
    fig.update_layout(
        title="<b>Factory Wide Energy Consumption</b>",
        xaxis_title="<b>Time (hours)</b>",
        yaxis_title="<b>Electricity Comsumption (kwH)</b>",
        height=500,
        margin=dict(l=80, r=20, t=50, b=60),
        # yaxis_title="Y Axis Title",
        legend_title="Forecasted Usage",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            xanchor="right",
            y=-0.2,
            x=1
        ),
        font=dict(
            size=12,
            color="White"
        ), 
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    fig.update_yaxes(automargin=True)

    metricsJSON = json.dumps(fig, cls = plotly.utils.PlotlyJSONEncoder)
    # fig.show()
    # fig = px.line(np.arange(1, len(data)+1), x='time', y="GOOG")
    # fig.show()
    return costJSON1, metricsJSON


def wheel_graph(data):
    num_vertices = data

    if not isinstance(data, int):
        num_vertices = len(data)

    print(num_vertices)
    
    G = nx.complete_graph(num_vertices)

    return G, G.edges(), data


def plotly_wheel_graph(G, edges, data):
    num_vertices = data

    if not isinstance(data, int):
        num_vertices = len(data)
    G = nx.complete_graph(num_vertices)
    # G.add_edges_from(edges)
    pos = nx.spring_layout(G)
    for n1,n2,attr in G.edges(data=True):
        print (n1,n2,attr)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_adjacencies = []
    node_text = []
    count = 0
    for node, adjacencies in enumerate(G.adjacency()):

        name = "Starting Endpoint"
        sensor_name = "Starting Sensor"

        if not isinstance(data, int):
            name = data[count][0]
            sensor_name = data[count][1]

        count += 1
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append('# of connections:' + str(len(adjacencies[1])) + ', Name: ' + str(name) + ', Sensor: ' + str(sensor_name))

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    # text="Node Graph Display",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )


    graph1JSON = json.dumps(fig, cls = plotly.utils.PlotlyJSONEncoder)
    return graph1JSON
  


def plotly_empty_graph(G, edges, data):
    G.remove_edges_from(list(G.edges))
    pos = nx.spring_layout(G)
    for n1,n2,attr in G.edges(data=True):
        print (n1,n2,attr)


    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append('# of connections: ' + str(len(adjacencies[1])))

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[node_trace],
             layout=go.Layout(
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    # text="Node Graph Display",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )


    graph1JSON = json.dumps(fig, cls = plotly.utils.PlotlyJSONEncoder)
    return graph1JSON
