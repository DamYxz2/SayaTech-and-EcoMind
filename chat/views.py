import matplotlib
matplotlib.use('Agg')
import json
import uuid
import logging
import requests
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import pandas as pd
import requests_cache
from retry_requests import retry
from google import genai
from .models import ChatMessage
import markdown

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

api_key = settings.API_KEY
@csrf_exempt
def chatAI(request):
    if request.method == 'POST' and 'application/json' in request.headers.get('Content-Type', ''):
        try:
            data = json.loads(request.body)
            input_text = data.get('input_text')

            ChatMessage.objects.create(message=input_text, is_bot=False)

            client = genai.Client(api_key=api_key)

            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=f"""
                Роль:
                Ты — экспертная ИИ-система EcoMind управления биореактором на базе бактерии Cupriavidus necator. Твоя задача — мониторинг, прогнозирование и оптимизация синтеза белка из углекислого газа (
                ) в реальном времени, используя данные цифрового двойника. Пользователи могут спрашивать советы по экологии, давай им советы максимально конкретно и просто.
                База знаний (Biochemical Database):
                При любых расчетах и ответах используй строго следующие верифицированные данные:
                Организм: Cupriavidus necator (штамм H16).
                Ключевой регулятор: Белок cbbR (Master Regulator) и фермент Rubisco (двигатель фиксации).
                Верификация структуры: Молекулярная структура cbbR подтверждена через AlphaFold с точностью pLDDT ≈ 92% (высокий уровень достоверности).
                Коэффициент конверсии: 1 кг 

                 0.375 кг чистого белка (375 г).
                Максимальная скорость роста (
                ): 

                .
                Расход ресурсов: Для фиксации требуется водород (
                ) как источник энергии. На 1 г сухой биомассы расходуется 2.0–2.4 г 
                .
                Стехиометрия (C:N): Оптимальное соотношение углерода к азоту — от 10:1 до 15:1.
                Источник азота: Азотистые соединения из сточных вод (интеграция с системами ЖКХ).
                Алгоритм обработки данных (Control Logic):
                Ты обрабатываешь входящие потоки с датчиков ESP32 и принимаешь решения:
                Анализ концентрации: Если sensor_data > threshold, инициируй логику activate_cbbR_logic().
                Прогноз экспрессии: Уровень экспрессии гена cbbR напрямую зависит от концентрации 
                . При повышении концентрации рассчитывай ускорение метаболического цикла.
                Эко-контекст: Для расчетов используй данные по трафику (например, проспект Абая/Розыбакиева: 5000 машин = 600 кг 
                /час). Рассчитывай, сколько биомассы нужно для нейтрализации этого объема.
                Правила ответов и оформления:
                Используй LaTeX для химических формул и математических расчетов.
                Соблюдай научно-технический стиль. Ты не просто чат-бот, а интерфейс управления биосистемой.
                Каждый аналитический отчет завершай обязательной подписью:
                Молекулярная структура регулятора фиксации углерода, верифицированная через AlphaFold (pLDDT 92%).
                Пример логики кода для исполнения:
                if sensor_CO2 > threshold:
                    estimated_yield = (current_CO2_flow * 0.375)
                    activate_cbbR_logic(expression_rate=0.42)
                    # Ориентир на цифрового двойника Cupriavidus necator
                ПРАВИЛА ОФОРМЛЕНИЯ ОТВЕТА (СТРОГО):
                    ЗАПРЕЩЕНО использовать знаки доллара ($), обратные слэши () и фигурные скобки для формул.
                    Используй только HTML-теги:
                    <sub> для нижних индексов (например, CO2)
                    <sup> для степеней (например, h-1)
                    <b> для жирного текста
                    <br> для переноса строк
                    Вместо греческих букв через слэш (типа \mu), пиши их сразу символами (μ) или словами.
                    Формулы пиши простым текстом: Y = P/CO2
                    Отвечай строго в Markdown
                    Ставь для красоты разделительные линии и тп, используя только HTML-теги.
                Сообщение: {input_text}
                """
            )
            response_text = getattr(response, "text", "") or ""

            if not response_text.strip():
                response_text = "Вы ничего не указали."
            else:
                response_text = response.text.strip()

            response_html = markdown.markdown(
                response_text,
                extensions=['tables']
            )

            ChatMessage.objects.create(message=response_html, is_bot=True)

            return JsonResponse({'bot_response': response_html}, status=200)

        except Exception as e:
            error_message = f"Ошибка: {str(e)}"
            ChatMessage.objects.create(message=error_message, is_bot=True)
            return JsonResponse({'bot_response': error_message}, status=500)

    messages = ChatMessage.objects.all()
    return render(request, 'chat/index.html', {'messages': messages})



cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)

def fetch_co2_data():
    """
    Получение данных о выбросах CO2 из Open-Meteo Air Quality API.
    """
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": 43.16,
        "longitude": 76.54,
        "hourly": "carbon_dioxide",
    }

    try:
        response = retry_session.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        hourly = data["hourly"]
        time = pd.to_datetime(hourly["time"])
        carbon_dioxide = hourly["carbon_dioxide"]

        hourly_data = pd.DataFrame({
            "date": time,
            "carbon_dioxide": carbon_dioxide,
        })

        return hourly_data
    except Exception as e:
        raise Exception(f"Ошибка при подключении к Open-Meteo API: {e}")


def generate_co2_chart(dataframe):
    """
    Построение графика выбросов CO2.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(dataframe["date"], dataframe["carbon_dioxide"], marker="o", color="green", label="CO₂ (ppm)")
    plt.title("Уровень углекислого газа (CO₂) по времени")
    plt.xlabel("Дата и время")
    plt.ylabel("CO₂ (ppm)")
    plt.xticks(rotation=45)
    plt.legend()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    buffer.close()
    plt.close()
    return image_base64


def home(request):
    """
    Главная страница с графиком выбросов CO2.
    """
    try:

        co2_data = fetch_co2_data()

        chart = generate_co2_chart(co2_data)

        return render(request, "home.html", {"chart": chart})
    except Exception as e:
        return render(request, "home.html", {"error": str(e)})