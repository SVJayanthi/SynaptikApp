# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

import os

class Config(object):
    basedir = os.path.abspath(os.path.dirname(__file__))

    # Set up the App SECRET_KEY
    # SECRET_KEY = config('SECRET_KEY'  , default='S#perS3crEt_007')
    SECRET_KEY = os.getenv('SECRET_KEY', 'S#perS3crEt_007')

    # This will create a file in <app> FOLDER
    # SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'db.sqlite3')
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "sqlite://")
    SQLALCHEMY_TRACK_MODIFICATIONS = False 

    # Assets Management
    ASSETS_ROOT = os.getenv('ASSETS_ROOT', 'apps/static/assets') 
    APP_ROOT = os.getenv('APP_ROOT', 'apps') 
    
class ProductionConfig(Config):
    DEBUG = False

    # Security
    SESSION_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_DURATION = 3600

    # PostgreSQL database
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "sqlite://")

    # SQLALCHEMY_DATABASE_URI = '{}://{}:{}@{}:{}/{}'.format(
    #     os.getenv('DB_ENGINE'   , 'postgresql'),
    #     os.getenv('DB_USERNAME' , 'hello_flask'),
    #     os.getenv('DB_PASS'     , 'hello_flask'),
    #     os.getenv('DB_HOST'     , 'db'),
    #     os.getenv('DB_PORT'     , 5432),
    #     os.getenv('DB_NAME'     , 'hello_flask_prod')
    # ) 

class DebugConfig(Config):
    DEBUG = True


# Load all possible configurations
config_dict = {
    'Production': ProductionConfig,
    'Debug'     : DebugConfig
}
