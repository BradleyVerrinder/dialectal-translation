from django import forms
from .models import User
from django.contrib.auth.forms import UserCreationForm


class CustomUserCreationForm(UserCreationForm):
    email = forms.EmailField(required=True, help_text="Enter your email address")
    class Meta(UserCreationForm.Meta):
        model = User
        fields = UserCreationForm.Meta.fields + ('email',)

class LoginForm(forms.Form):
    username = forms.CharField(max_length=150)
    password = forms.CharField(widget=forms.PasswordInput)