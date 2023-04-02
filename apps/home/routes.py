# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""
from __future__ import absolute_import

from apps.home import blueprint
from flask import render_template, request
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

import sys

sys.path.insert(0,'CityLearn')

from CityLearn.citylearn.citylearn import CityLearnEnv
# from citylearn import agents

from CityLearn.citylearn.agents.marlisa import MARLISA
from CityLearn.citylearn.agents.rbc import BasicRBC
# from citylearn import CityLearnEnv
from apps.citylearn.helpers import *
from apps.citylearn.im_save import save_image
from PIL import Image
import numpy as np
from multiprocessing import Process


plt.switch_backend('agg')


env = CityLearnEnv(schema=Constants.schema_path)
env.reset()
@blueprint.route('/home')
# @login_required
def home():
    app_config = current_app.config

    comm_graph, metric_values = env.render_comm()

    comm_graph_dir = save_image(app_config, comm_graph, name="comm_graph")
    # metric_chart_dir = save_image(app_config, comm_graph, name="metric_chart")
    costJSON, metricsJSON = plotly_data_vis(metric_values)

    graphJSON = plotly_wheel_graph(1)
    
    counter = session.get('graphJSON', graphJSON)
    counter = session.get('costJSON', costJSON)
    counter = session.get('metricsJSON', metricsJSON)

    return render_template('home/test.html', title = "Home", graphJSON=graphJSON, comm_graph = comm_graph_dir, metric_chart = metricsJSON, cost_chart = costJSON)
    # return render_template('home/test.html', title = "Home")

def simulation(app_config):
    done = False
    obs = env.reset()
    steps = 0
    while not done and steps < 10:
        actions = [a.sample() for a in env.action_space]

        # apply actions to citylearn_env
        next_observations, rew, done, _ = env.step(actions)

        observations = [o for o in next_observations]

        comm_graph, metric_values = env.render_comm()
        comm_graph_dir = save_image(app_config, comm_graph, name="comm_graph")

        steps += 1

def start_background_process(app_config):
    process = Process(target=simulation,  args=(app_config, ))
    process.daemon = True  # Set the process as a daemon process so that it automatically stops when the main process exits
    process.start()

# @app.route('/start-background-routine', methods=['POST'])
@blueprint.route('/simulation',  methods=['POST'])
def start_simulation():
    app_config = current_app.config
    if request.method == 'POST':
        start_background_process(app_config)
        return "Background routine started."
    
@blueprint.route('/update_data',  methods=['POST'])
def update_data():
    app_config = current_app.config
    if request.method == 'POST':
        graphJSON = session.get('graphJSON')
        costJSON = session.get('costJSON')
        metricsJSON = session.get('metricsJSON')
        return jsonify({'graphJSON': graphJSON, 'costJSON': costJSON, 'metricsJSON': metricsJSON})
    # app_config = current_app.config

    # graph1JSON = plotly_wheel_graph(5)

    # comm_graph, metric_values = env.render_comm()

    # comm_graph_dir = save_image(app_config, comm_graph, name="comm_graph")
    # costJSON, metricsJSON = plotly_data_vis(metric_values)

    # done = False
    # obs = env.reset()
    # # rewards = [[] for _ in range(5)]
    # steps = 0
    # imgs = []
    # while not done and steps < 10:
    #     actions = [a.sample() for a in env.action_space]

    #     # apply actions to citylearn_env
    #     next_observations, rew, done, _ = env.step(actions)

    #     observations = [o for o in next_observations]

    #     comm_graph, metric_values = env.render_comm()
    #     comm_graph_dir = save_image(app_config, comm_graph, name="comm_graph")
    #     costJSON, metricsJSON =  plotly_data_vis(metric_values)

    #     steps += 1

    # return render_template('home/test.html', title = "Home", graph1JSON=graph1JSON, comm_graph = comm_graph_dir, metric_chart = metricsJSON, cost_chart = costJSON)

@blueprint.route('/go')
# @login_required
def graph_index():

    return render_template('graphs/index.html', segment='graph_index')


@blueprint.route('/<template>')
@login_required
def route_template(template):

    try:

        if not template.endswith('.html'):
            template += '.html'

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/home/FILE.html
        return render_template("home/" + template, segment=segment)

    except TemplateNotFound:
        return render_template('home/page-404.html'), 404

    except:
        return render_template('home/page-500.html'), 500


# Helper - Extract current page name from request
def get_segment(request):

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None


@blueprint.route('/graph',  methods=['POST'])
# @login_required
def generate_graph():
    url = request.form['url']

    data = int(url)
    graph1JSON = plotly_wheel_graph(data)

    return render_template('home/test.html', title = "Home", graph1JSON = graph1JSON)

    # return render_template('home/index.html', graph=prefix+new_graph_name)

def plotly_data_vis(data):
    names = ['Standard', 
            'Devices without Storage', 
            'Devices without Storage & PV']
    
    time_of_day = 0.004*np.random.rand(len(data[0]))
    cumulative_cost = np.cumsum(np.clip(data[0], 0, None))*time_of_day
    df = pd.DataFrame({'Energy Expenditures': cumulative_cost, 'Time of Day Rate': time_of_day})
    # df = pd.DataFrame([[], [time_of_day]], columns=["Cumulative Expenditures", "Time of Day Rate"])
    
    fig = px.line(df)

    fig.update_layout(
        title="Factory Wide Energy Expenditures",
        xaxis_title="Time (sec)",
        yaxis_title="Cost ($)",
        height=400,
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
            color="Black"
        ), 
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
        title="Factory Wide Energy Consumption",
        xaxis_title="Time (sec)",
        yaxis_title="Electricity Comsumption (kwH)",
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
            color="Black"
        ), 
    )
    
    fig.update_yaxes(automargin=True)

    metricsJSON = json.dumps(fig, cls = plotly.utils.PlotlyJSONEncoder)
    # fig.show()
    # fig = px.line(np.arange(1, len(data)+1), x='time', y="GOOG")
    # fig.show()
    return costJSON1, metricsJSON


def plotly_wheel_graph(data):
    
    G = nx.wheel_graph(data)

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
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append('# of connections: '+str(len(adjacencies[1])))

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Node Graph Display",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )


    graph1JSON = json.dumps(fig, cls = plotly.utils.PlotlyJSONEncoder)
    return graph1JSON
  
# app.run(host='0.0.0.0', port=81, debug=True)
