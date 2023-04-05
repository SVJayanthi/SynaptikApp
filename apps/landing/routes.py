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
from flask_login import (
    current_user,
    login_user,
    logout_user
)
from apps.authentication.models import Users

from apps import db, login_manager
plt.switch_backend('agg')


@blueprint.route('/index')
# @login_required
def index():
    return render_template('landing/index.html', title = "Landing")


@blueprint.route('/contact', methods=['POST'])
# @login_required
def contact():
    
    name = request.form['name']
    email = request.form['email']
    message = request.form['message']
    return render_template('landing/index.html', title = "Landing")