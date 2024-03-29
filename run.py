# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

import os
from   flask_migrate import Migrate
from   flask_minify  import Minify
from   sys import exit
from flask import render_template, request, current_app

from apps.config import config_dict
from apps import create_app, db

from turbo_flask import Turbo
from threading import Thread
import time


# WARNING: Don't run with debug turned on in production!
DEBUG = (os.getenv('DEBUG', 'False') == 'True')

# The configuration
get_config_mode = 'Debug' if DEBUG else 'Production'

try:
    # print(DEBUG)
    # print(dir(config_dict['Debug']))
    # Load the configuration using the default values
    app_config = config_dict[get_config_mode.capitalize()]

except KeyError:
    exit('Error: Invalid <config_mode>. Expected values [Debug, Production] ')


# APP_ROOT = os.path.dirname(os.path.abspath(__file__))
# UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static', 'customlogos')

app = create_app(app_config)
Migrate(app, db)

# turbo = Turbo(app)

if not DEBUG:
    Minify(app=app, html=True, js=False, cssless=False)
    
if DEBUG:
    app.logger.info('DEBUG            = ' + str(DEBUG)             )
    app.logger.info('FLASK_ENV        = ' + os.getenv('FLASK_ENV') )
    app.logger.info('Page Compression = ' + 'FALSE' if DEBUG else 'TRUE' )
    app.logger.info('DBMS             = ' + app_config.SQLALCHEMY_DATABASE_URI)
    app.logger.info('ASSETS_ROOT      = ' + app_config.ASSETS_ROOT )


if __name__ == "__main__":
    app.run()
