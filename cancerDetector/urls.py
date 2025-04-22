from django.contrib import admin
from django.urls import path
from . import views  # importa tu archivo views.py

urlpatterns = [
    #path('admin/', admin.site.urls),
    path('', views.homepage, name='homepage'),
]
