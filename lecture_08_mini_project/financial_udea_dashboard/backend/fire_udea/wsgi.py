"""
WSGI config for fire_udea project.
"""
import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fire_udea.settings')
application = get_wsgi_application()
