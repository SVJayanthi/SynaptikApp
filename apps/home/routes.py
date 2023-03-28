# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from apps.home import blueprint
from flask import render_template, request
from flask_login import login_required
from jinja2 import TemplateNotFound
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


plt.switch_backend('agg')


@blueprint.route('/index')
# @login_required
def index():
    return render_template('landing/index.html', title = "Landing")

@blueprint.route('/home')
# @login_required
def home():

    df = px.data.medals_wide()
    fig1 = px.bar(df, x = "nation", y = ['gold', 'silver', 'bronze'], title = "Wide=FormInput")

    graph1JSON = json.dumps(fig1, cls = plotly.utils.PlotlyJSONEncoder)

    return render_template('home/test.html', title = "Home", graph1JSON = graph1JSON)
    # return render_template('home/test.html', title = "Home")


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

    data=int(url)
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

    return render_template('home/test.html', title = "Home", graph1JSON = graph1JSON)

    # return render_template('home/index.html', graph=prefix+new_graph_name)

  
# app.run(host='0.0.0.0', port=81, debug=True)
