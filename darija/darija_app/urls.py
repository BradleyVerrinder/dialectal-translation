from django.urls import path
from darija_app import views

urlpatterns = [
    path("", views.home, name="home"),
    path("home", views.home, name="home"),
    path("login", views.login, name="login"),
    path("logout", views.logout, name="logout"),
    path("registration", views.registration, name="registration"),
    path("translate_admin", views.translate_admin, name="translate_admin"),
    path('add_to_favourites', views.add_to_favourites, name='add_to_favourites'),
    path('delete_favourite/<int:id>/', views.delete_favourite, name='delete_favourite'),
]