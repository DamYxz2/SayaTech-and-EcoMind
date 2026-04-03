from django.urls import path
from .views import *

urlpatterns = [
    path("", home, name="home"),
    path("chat/", index, name="chat"),
    path("chatAI/", chatAI, name="api_chat"),
]
