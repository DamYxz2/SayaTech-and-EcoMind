import json
import uuid
import logging

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from chat.models import Conversation, Message

logger = logging.getLogger(__name__)

_engine = None

def get_engine():
    global _engine
    if _engine is None:
        from ai_engine.inference import EcoMindEngine
        checkpoint_dir = settings.ECOMIND_CHECKPOINT_DIR
        logger.info(f"Загрузка EcoMind AI из {checkpoint_dir}...")
        _engine = EcoMindEngine(checkpoint_dir)
        logger.info("EcoMind AI загружен!")
    return _engine

def index(request):
    return render(request, "chat/index.html")

@csrf_exempt
@require_http_methods(["POST"])
def api_chat(request):
    try:
        data = json.loads(request.body)
        user_message = data.get("message", "").strip()
        session_id = data.get("session_id", str(uuid.uuid4()))

        if not user_message:
            return JsonResponse({"error": "Пустое сообщение"}, status=400)

        conversation, _ = Conversation.objects.get_or_create(session_id=session_id)

        Message.objects.create(
            conversation=conversation,
            role="user",
            text=user_message,
        )

        engine = get_engine()
        result = engine.answer(user_message)

        Message.objects.create(
            conversation=conversation,
            role="bot",
            text=result["response"],
            intent=result["intent"],
            confidence=result["confidence"],
        )

        return JsonResponse(
            {
                "response": result["response"],
                "intent": result["intent"],
                "confidence": round(result["confidence"], 3),
                "method": result["method"],
                "session_id": session_id,
            }
        )

    except json.JSONDecodeError:
        return JsonResponse({"error": "Невалидный JSON"}, status=400)
    except Exception as e:
        logger.exception("Ошибка в api_chat")
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
@require_http_methods(["GET"])
def api_history(request, session_id):
    try:
        conversation = Conversation.objects.get(session_id=session_id)
        messages = conversation.messages.all().values(
            "role", "text", "intent", "confidence", "created_at"
        )
        return JsonResponse({"messages": list(messages)})
    except Conversation.DoesNotExist:
        return JsonResponse({"messages": []})
