from django.shortcuts import redirect
from django.contrib import messages
from django.http import HttpResponse


def custom_login_required(view_func):
    def wrapper(request, *args, **kwargs):
        if request.user.is_authenticated:
            return view_func(request, *args, **kwargs)
        else:
            messages.error(request, "You must log in to access the dashboard.")
            return redirect('login')
    return wrapper