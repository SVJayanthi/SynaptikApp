# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from flask import Flask
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
from importlib import import_module

# Import Dash application
from apps.dashboard import DashApp, DashApp1, DashApp2

global_dict = {"communication": True, 'num_vertices': 1}

db = SQLAlchemy()
login_manager = LoginManager()
dash_class = DashApp(global_dict)
dash_class1 = DashApp1(global_dict)
dash_class2 = DashApp2(global_dict)

def register_extensions(app):
    db.init_app(app)
    login_manager.init_app(app)


def register_blueprints(app):
    for module_name in ('authentication', 'home', 'landing'):
        module = import_module('apps.{}.routes'.format(module_name))
        app.register_blueprint(module.blueprint)


def configure_database(app):

    @app.before_first_request
    def initialize_database():
        db.create_all()

    @app.teardown_request
    def shutdown_session(exception=None):
        db.session.remove()


def create_app(config):
    app = Flask(__name__)
    app.config.from_object(config)
    # print(app.config)
    register_extensions(app)
    app = dash_class.init_dashboard(app)
    app = dash_class1.init_dashboard(app)
    app = dash_class2.init_dashboard(app)

    register_blueprints(app)
    configure_database(app)
    return app
