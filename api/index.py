from app import app
from vercel_wsgi import handle_request

def handler(environ, start_response):
    return handle_request(app, environ, start_response) 