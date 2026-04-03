from django.db import models


class Conversation(models.Model):
    """Сессия разговора"""
    session_id = models.CharField(max_length=64, unique=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Conversation {self.session_id}"


class Message(models.Model):
    """Одно сообщение в чате"""
    ROLE_CHOICES = [
        ("user", "Пользователь"),
        ("bot", "Бот"),
    ]

    conversation = models.ForeignKey(
        Conversation, on_delete=models.CASCADE, related_name="messages"
    )
    role = models.CharField(max_length=4, choices=ROLE_CHOICES)
    text = models.TextField()
    intent = models.CharField(max_length=64, blank=True, null=True)
    confidence = models.FloatField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"[{self.role}] {self.text[:50]}"
