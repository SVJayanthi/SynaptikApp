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
import os
import time
import plotly.express as px
import pandas as pd
import plotly

plt.switch_backend('agg')


@blueprint.route('/index')
# @login_required
def index():
    prefix = 'assets/img/'
    new_graph_name = "graph.png"

    df = px.data.medals_wide()
    fig1 = px.bar(df, x = "nation", y = ['gold', 'silver', 'bronze'], title = "Wide=FormInput")

    graph1JSON = json.dumps(fig1, cls =plotly.utils.PlotlyJSONEncoder)

    return render_template('home/test.html', title = "Home", graph1JSON = graph1JSON)


@blueprint.route('/graph_index')
# @login_required
def graph_index():

    return render_template('graphs/index.html', segment='graph_index')