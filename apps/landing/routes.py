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
    return render_template('landing/index.html', title = "Landing")