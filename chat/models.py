from django.db import models

# Create your models here.
class ChatMessage(models.Model):
    message = models.TextField()
    is_bot = models.BooleanField(default=False)
    def __str__(self):
        return self.message