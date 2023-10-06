from django.urls import path
from FaceApp import views

urlpatterns = [
    path('', views.index2, name='index2')
]