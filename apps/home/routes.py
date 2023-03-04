# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from apps.home import blueprint
from flask import render_template, request
from flask_login import login_required
from jinja2 import TemplateNotFound
import requests
import json
import networkx as nx
import matplotlib.pyplot as plt
import os
import time


@blueprint.route('/index')
# @login_required
def index():

    return render_template('home/index.html', segment='index')


@blueprint.route('/graph_index')
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



# @blueprint.route('/index')
# @login_required
# def index():
#     return render_template('index.html')


@blueprint.route('/graph',  methods=['POST'])
# @login_required
def generate_graph():
    # url = requests.form['url']
    # response = requests.get(url)
    # data = json.loads(response.content)
    # var = request.form["samplename"]
    graph = nx.wheel_graph(7)
    # graph = nx.Graph()
    # for node in data['nodes']:
    #     graph.add_node(node['id'], label=node['label'])
    # for edge in data['edges']:
    #     graph.add_edge(edge['source'], edge['target'])
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos)
    nx.draw_networkx_edges(graph, pos)
    nx.draw_networkx_labels(graph, pos, labels=nx.get_node_attributes(graph, 'label'))
    plt.axis('off')
    # plt.savefig('static/assets/img/graph.png')

    print(os.listdir())
    static = 'apps/static/'
    prefix = 'assets/img/graph/'
    new_graph_name = "graph_" + str(time.time()) + ".png"

    for filename in os.listdir(static+prefix):
        if filename.startswith('graph_'):  # not to remove other images
            os.remove( static + prefix + filename)

    plt.savefig(static+ prefix + new_graph_name)

    return render_template('home/index.html', graph=prefix+new_graph_name)

  
# app.run(host='0.0.0.0', port=81, debug=True)
