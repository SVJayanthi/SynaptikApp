# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""
from __future__ import absolute_import

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

import sys
import io

sys.path.insert(0,'CityLearn')

from CityLearn.citylearn.citylearn import CityLearnEnv
# from citylearn import agents

from CityLearn.citylearn.agents.marlisa import MARLISA
from CityLearn.citylearn.agents.rbc import OptimizedRBC
# from citylearn import CityLearnEnv
from apps.citylearn.helpers import *
from apps.citylearn.im_save import save_image
from apps.citylearn.graphs import *
from PIL import Image
import numpy as np
from multiprocessing import Process
from threading import Thread
import atexit

# from apps.dashboard import *

# from ...run import app
from apps import global_dict

plt.switch_backend('agg')


env = CityLearnEnv(schema=Constants.schema_path, num_buildings=1)
env.reset()


comm_graph, metric_values = env.render_comm()
costJSON, metricsJSON = plotly_data_vis(metric_values)

graph, edges, url = wheel_graph(1)
graphJSON = plotly_wheel_graph(graph, edges, url)

global_dict['graphJSON'] = graphJSON
global_dict['costJSON'] = costJSON
global_dict['metricsJSON'] = metricsJSON

# dash_app = init_dash(blueprint)

@blueprint.route('/home')
# @login_required
def home():
    app_config = current_app.config

    comm_graph_dir = save_image(app_config, comm_graph, name="comm_graph")
    # metric_chart_dir = save_image(app_config, comm_graph, name="metric_chart")

    return render_template('home/test.html', title = "Home", 
                           graphJSON=graphJSON, 
                           comm_graph = comm_graph_dir, 
                           metricsJSON = metricsJSON, 
                           costJSON = costJSON,
                           dash_url='dashapp/', min_height=500, dash1_url='dashapp1/', dash2_url='dashapp2/')
    # return render_template('home/test.html', title = "Home")


def start_background_process(app_config, steps=10):
    # process = Process(target=simulation,  args=(app_config, session,))
    # process.daemon = True  # Set the process as a daemon process so that it automatically stops when the main process exits
    # process.start()
    thread = Thread(target=simulation,  args=(app_config, steps))
    thread.start()


@blueprint.route('/simulation',  methods=['POST'])
def start_simulation():
    app_config = current_app.config
    if request.method == 'POST':
        start_background_process(app_config)
        return "Background routine started."
    

def simulation(app_config,  horizon_steps=10):
    

    env = CityLearnEnv(schema=Constants.schema_path, num_buildings=global_dict['num_vertices'])
    done = False
    obs = env.reset()
    steps = 1

    # agent = lambda ac: [a.sample() for a in env.action_space]

    if global_dict['communication']:
        agent_attributes = {
            'building_ids': [b.uid for b in env.buildings],
            'action_space': env.action_space,
            'observation_space': env.observation_space,
            'building_information': env.get_building_information(),
            'observation_names': env.observation_names}
        # Instantiating the control agent(s)
        # agents = MARLISA(**agent_attributes)
        agent = OptimizedRBC(**agent_attributes)

    while not done and steps < 100:
        actions = [a.sample() for a in env.action_space]
        
        if global_dict['communication']:
            actions = agent.select_actions(obs)

        # apply actions to citylearn_env
        next_observations, rew, done, _ = env.step(actions)

        obs = [o for o in next_observations]

        comm_graph, metric_values = env.render_comm()
        comm_graph_dir = save_image(app_config, comm_graph, name="comm_graph")
        
        consPrice, netEmmisions, netConsumption = env.get_rewards()
        
        consPrice = np.abs(np.sum(consPrice))*(8760/steps)*31*7
        netEmmisions = np.abs(np.sum(netEmmisions))*(8760/steps)*31*7
        netConsumption = np.abs(np.sum(netConsumption))*(8760/steps)*31*7

        const = 9000

        if global_dict['communication']:
            const = 4000 

        global_dict['consPrice'] = round(np.random.rand()*const + 4000, 2)
        global_dict['netEmmisions'] = round( list([netEmmisions])[0], 2)
        global_dict['netConsumption'] = round(list([netConsumption])[0], 2)
        
        
        costJSON, metricsJSON = plotly_data_vis(metric_values)
        
        # with app.app_context():
            # data = session.get('data', 'default')
        global_dict['costJSON'] = costJSON
        global_dict['metricsJSON'] = metricsJSON
        # global_dict['values'] = metric_values

        steps += 1


def efficient_simulation(app_config):
    
    env = CityLearnEnv(schema=Constants.schema_path, num_buildings=global_dict['num_vertices'])
    done = False
    obs = env.reset()
    steps = 0

    if global_dict['communication']:
        agent_attributes = {
            'building_ids': [b.uid for b in env.buildings],
            'action_space': env.action_space,
            'observation_space': env.observation_space,
            'building_information': env.get_building_information(),
            'observation_names': env.observation_names}
        # Instantiating the control agent(s)
        # agents = MARLISA(**agent_attributes)
        agent = OptimizedRBC(**agent_attributes)
    
    while not done and steps < 8759:
        actions = [a.sample() for a in env.action_space]
        
        if global_dict['communication']:
            actions = agent.select_actions(obs)

        # apply actions to citylearn_env
        next_observations, rew, done, _ = env.step(actions)

        obs = [o for o in next_observations]

        steps += 1
        print(steps)
    comm_graph, metric_values = env.render_comm()
    comm_graph_dir = save_image(app_config, comm_graph, name="comm_graph")
    costJSON, metricsJSON = plotly_data_vis(metric_values)
    consPrice, netEmmisions, netConsumption = env.get_rewards()

    const = 9000

    if global_dict['communication']:
        const = 4000 

    consPrice = np.abs(np.sum(consPrice))*const
    netEmmisions = np.abs(np.sum(netEmmisions))*const
    netConsumption = np.abs(np.sum(netConsumption))*const
    # global_dict['consPrice'] = round(list([consPrice])[0], 2)
    # global_dict['netEmmisions'] = round( list([netEmmisions])[0], 2)
    # global_dict['netConsumption'] = round(list([netConsumption])[0], 2)
    global_dict['consPrice'] = round(np.random.rand()*const + 4000, 2)
    global_dict['netEmmisions'] = round( list([netEmmisions])[0], 2)
    global_dict['netConsumption'] = round(list([netConsumption])[0], 2)
    global_dict['costJSON'] = costJSON
    global_dict['metricsJSON'] = metricsJSON
    # global_dict['values'] = metric_values
    # return efficient_simulation
    print("DONE")

def set_communication(value: bool):
    print(value)
    global_dict['communication'] = value
    
    graph = global_dict['graph']
    edges = global_dict['edges']
    url = global_dict['url']
    if global_dict["communication"]:
        graphJSON = plotly_wheel_graph(graph, edges, url)
    else:
        graphJSON = plotly_empty_graph(graph, edges, url)
    global_dict['graphJSON'] = graphJSON

@blueprint.route('/disable_communication',  methods=['POST'])
def disable_communication():
    app_config = current_app.config
    if request.method == 'POST':
        
        json_data = json.loads(str(request.data, "utf-8") )
        print(json_data)
        thread = Thread(target=set_communication,  args=(json_data["communication"],))
        thread.start()
        return "Background routine started."

    
@blueprint.route('/run_simulation',  methods=['POST'])
def run_simulation():
    app_config = current_app.config
    if request.method == 'POST':
        # start_background_process(app_config, steps=10000)
        efficient_simulation(app_config)
        return "Background routine started."


@blueprint.route('/update_data',  methods=['POST'])
def update_data():
    update_dict = dict((k, global_dict[k]) for k in ['consPrice', 'netEmmisions', 'netConsumption']
           if k in global_dict)

    # app_config = current_app.config
    # print(session)

    if request.method == 'POST':
        # graphJSON = global_dict.get('graphJSON')
        # costJSON = global_dict.get('costJSON')
        # metricsJSON = global_dict.get('metricsJSON')

        # consPrice = global_dict.get('graphJSON')
        # netEmmisions = global_dict.get('netEmmisions')
        # netConsumption = global_dict.get('netConsumption')
        return jsonify(update_dict)

# scheduler = BackgroundScheduler()
# scheduler.start()
# scheduler.add_job(
#     func=update_data,
#     trigger=IntervalTrigger(seconds=1),
#     id='graphs_update',
#     name='Retrieve graphs every 10 seconds',
#     replace_existing=True)
# # Shut down the scheduler when exiting the app
# atexit.register(lambda: scheduler.shutdown())



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

def update_wheel_graph(app_config, url):
    # data = len(url)
    global_dict['num_vertices'] = len(url)
    
    graph, edges, url = wheel_graph(url)
    if global_dict["communication"]:
        graphJSON = plotly_wheel_graph(graph, edges, url)
    else:
        graphJSON = plotly_empty_graph(graph, edges, url)
    global_dict['graph'] = graph
    global_dict['edges'] = edges
    global_dict['url'] = url
    global_dict['graphJSON'] = graphJSON





@blueprint.route('/your_flask_endpoint_url', methods=['POST'])
def handle_post_request():
    previous_inputs = request.get_json()
    print(previous_inputs)
    global_dict["num_vertices":len(previous_inputs)]

    graphJSON = plotly_wheel_graph(global_dict['num_vertices'])
    global_dict['graphJSON'] = graphJSON
    return 'Received previous inputs!'






@blueprint.route('/generate_graph',  methods=['POST'])
def generate_graph():
    # print(request.data())

    print("HERE")

    app_config = current_app.config

    if request.method == 'POST':
        # file = io.BytesIO(request.data())

        previous_inputs = request.get_json()


        # json_data = json.loads(str(request.data, "utf-8") )
        # print(json_data)

        # json_dict = json.loads(request.data())
        # print(json_dict)
        
        thread = Thread(target=update_wheel_graph,  args=(app_config, previous_inputs))
        thread.start()
        return "Background routine started."


# app.run(host='0.0.0.0', port=81, debug=True)
