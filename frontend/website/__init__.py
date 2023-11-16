from flask import Flask

def create_app():
    app =  Flask(__name__)

    from .views import views
    from .marking import marking
    from .marking_ai import marking_ai

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(marking, url_prefix='/')
    app.register_blueprint(marking_ai, url_prefix='/')

    return app
