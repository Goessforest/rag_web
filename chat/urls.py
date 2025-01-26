# chat/urls.py
from django.urls import path
from . import views
# import logging
# from django.conf import settings
# from django.conf.urls.static import static

urlpatterns = [
    path('', views.chat_home, name='chat_home'),
]

# # This is crucial to serve media files during development
# logging.warning(settings.DEBUG)
# if settings.DEBUG:
#     urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)