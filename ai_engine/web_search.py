"""
EcoMind Web Search — поиск в интернете без API ключей.

Использует:
  1. Wikipedia API (бесплатный, без ключей)
  2. DuckDuckGo Instant Answer API (бесплатный)
  3. Парсинг Google результатов как fallback

Зависимости: pip install requests beautifulsoup4
"""

import re
import logging
import urllib.parse
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Таймаут для всех запросов
TIMEOUT = 8
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def search_wikipedia(query: str, lang: str = "ru") -> Optional[str]:
    """
    Поиск в Wikipedia API — полностью бесплатный, без ключей.
    Возвращает краткое описание статьи.
    """
    try:
        # 1. Поиск статьи
        search_url = f"https://{lang}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 3,
            "format": "json",
            "utf8": 1,
        }
        resp = requests.get(search_url, params=params, headers=HEADERS, timeout=TIMEOUT)
        data = resp.json()
        results = data.get("query", {}).get("search", [])

        if not results:
            return None

        # 2. Получаем содержимое лучшей статьи
        title = results[0]["title"]
        content_params = {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "exintro": True,        # только введение
            "explaintext": True,    # простой текст без HTML
            "exsectionformat": "plain",
            "format": "json",
            "utf8": 1,
        }
        resp2 = requests.get(search_url, params=content_params, headers=HEADERS, timeout=TIMEOUT)
        data2 = resp2.json()
        pages = data2.get("query", {}).get("pages", {})

        for page_id, page in pages.items():
            if page_id == "-1":
                continue
            extract = page.get("extract", "")
            if extract:
                # Обрезаем до разумной длины
                sentences = extract.split(". ")
                if len(sentences) > 6:
                    extract = ". ".join(sentences[:6]) + "."
                return f"**{title}** (Wikipedia):\n\n{extract}"

        return None

    except Exception as e:
        logger.warning(f"Wikipedia search error: {e}")
        return None


def search_duckduckgo(query: str) -> Optional[str]:
    """
    DuckDuckGo Instant Answer API — бесплатный, без ключей.
    Даёт краткие ответы на фактические вопросы.
    """
    try:
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
            "t": "ecomind_bot",
        }
        resp = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
        data = resp.json()

        # Проверяем разные поля ответа
        # Abstract — основное описание
        abstract = data.get("AbstractText", "")
        if abstract and len(abstract) > 50:
            source = data.get("AbstractSource", "")
            return f"**{source}:**\n\n{abstract}"

        # Answer — прямой ответ
        answer = data.get("Answer", "")
        if answer:
            return f"**Ответ:** {answer}"

        # Definition
        definition = data.get("Definition", "")
        if definition:
            return f"**Определение:** {definition}"

        # Related topics
        related = data.get("RelatedTopics", [])
        if related:
            texts = []
            for topic in related[:3]:
                text = topic.get("Text", "")
                if text:
                    texts.append(f"• {text}")
            if texts:
                return "**Найденная информация:**\n\n" + "\n".join(texts)

        return None

    except Exception as e:
        logger.warning(f"DuckDuckGo search error: {e}")
        return None


def search_google_scrape(query: str) -> Optional[str]:
    """
    Парсинг результатов Google — fallback вариант.
    Извлекает сниппеты из поисковой выдачи.
    """
    try:
        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://www.google.com/search?q={encoded_query}&hl=ru&num=5"

        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        soup = BeautifulSoup(resp.text, "html.parser")

        # Ищем блоки с результатами
        snippets = []

        # Featured snippet (блок с прямым ответом)
        featured = soup.select_one("div.hgKElc")
        if featured:
            return f"**Ответ из Google:**\n\n{featured.get_text(strip=True)}"

        # Обычные результаты
        for div in soup.select("div.tF2Cxc, div.g"):
            # Заголовок
            title_el = div.select_one("h3")
            # Сниппет
            snippet_el = div.select_one("div.VwiC3b, span.aCOpRe, div.IsZvec")

            if snippet_el:
                text = snippet_el.get_text(strip=True)
                if len(text) > 40:
                    title = title_el.get_text(strip=True) if title_el else ""
                    if title:
                        snippets.append(f"**{title}:**\n{text}")
                    else:
                        snippets.append(text)

            if len(snippets) >= 3:
                break

        if snippets:
            return "**Результаты поиска:**\n\n" + "\n\n".join(snippets)

        return None

    except Exception as e:
        logger.warning(f"Google scrape error: {e}")
        return None


def web_search(query: str) -> dict:
    """
    Главная функция поиска — пробует все источники по очереди.

    Returns:
        {
            "found": bool,
            "response": str,
            "source": str,  # "wikipedia", "duckduckgo", "google", "none"
        }
    """
    # 1. Wikipedia — самый надёжный
    result = search_wikipedia(query)
    if result:
        return {"found": True, "response": result, "source": "wikipedia"}

    # 2. DuckDuckGo — быстрые ответы
    result = search_duckduckgo(query)
    if result:
        return {"found": True, "response": result, "source": "duckduckgo"}

    # 3. Google — fallback
    result = search_google_scrape(query)
    if result:
        return {"found": True, "response": result, "source": "google"}

    return {
        "found": False,
        "response": "К сожалению, не удалось найти информацию по этому запросу. Попробуйте переформулировать вопрос.",
        "source": "none",
    }
