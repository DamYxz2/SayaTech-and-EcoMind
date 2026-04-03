from django.urls import path
from chat import views

urlpatterns = [
    path("", views.index, name="index"),
    path("api/chat/", views.api_chat, name="api_chat"),
    path("api/history/<str:session_id>/", views.api_history, name="api_history"),
]
