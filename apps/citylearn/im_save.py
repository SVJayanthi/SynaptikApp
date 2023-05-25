from __future__ import absolute_import

from apps.home import blueprint
from flask import render_template, request
from flask import url_for

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

df = px.data.stocks()
fig = px.line(df, x='date', y="GOOG")

import sys

sys.path.insert(0,'CityLearn')

from CityLearn.citylearn.citylearn import CityLearnEnv
# from citylearn import agents

from CityLearn.citylearn.agents.marlisa import MARLISA
from CityLearn.citylearn.agents.rbc import BasicRBC
# from citylearn import CityLearnEnv
from apps.citylearn.helpers import *
from PIL import Image
import numpy as np

imgs_subdir = '/img/'
# app_config = current_app.config

def white_to_transparency(img):
    x = np.asarray(img.convert('RGBA')).copy()

    x[:, :, 3] = (255 * (x[:, :, :3] != 255).any(axis=2)).astype(np.uint8)

    return Image.fromarray(x)


def save_image(app_config, arr, name):
    im = Image.fromarray(arr)
    
    # im = white_to_transparency(im)

    suffix = imgs_subdir + f'{name}.png'
    save_dir = app_config['ASSETS_ROOT'] + suffix
    # save_dir = app_config['APP_ROOT'] + app_config['ASSETS_ROOT'] + suffix
    im.save(save_dir)

    return 'assets' + suffix


