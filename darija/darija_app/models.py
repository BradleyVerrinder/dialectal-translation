from django.db import models
from django.contrib.auth.models import AbstractUser

# Custom User Model for Authentication
class User(AbstractUser):
    def __str__(self):
        return self.username
    
# Saved Translation Model
class Translation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='translations')
    source_text = models.TextField()
    translated_text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.source_text} - {self.translated_text}"