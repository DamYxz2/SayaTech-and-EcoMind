import re
import logging
import urllib.parse
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

TIMEOUT = 8
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

def search_wikipedia(query: str, lang: str = "ru") -> Optional[str]:
    try:
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

        title = results[0]["title"]
        content_params = {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
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
                sentences = extract.split(". ")
                if len(sentences) > 6:
                    extract = ". ".join(sentences[:6]) + "."
                return f"**{title}** (Wikipedia):\n\n{extract}"

        return None

    except Exception as e:
        logger.warning(f"Wikipedia search error: {e}")
        return None

def search_duckduckgo(query: str) -> Optional[str]:
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

        abstract = data.get("AbstractText", "")
        if abstract and len(abstract) > 50:
            source = data.get("AbstractSource", "")
            return f"**{source}:**\n\n{abstract}"

        answer = data.get("Answer", "")
        if answer:
            return f"**Ответ:** {answer}"

        definition = data.get("Definition", "")
        if definition:
            return f"**Определение:** {definition}"

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
    try:
        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://www.google.com/search?q={encoded_query}&hl=ru&num=5"

        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        soup = BeautifulSoup(resp.text, "html.parser")

        snippets = []

        featured = soup.select_one("div.hgKElc")
        if featured:
            return f"**Ответ из Google:**\n\n{featured.get_text(strip=True)}"

        for div in soup.select("div.tF2Cxc, div.g"):
            title_el = div.select_one("h3")
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
    result = search_wikipedia(query)
    if result:
        return {"found": True, "response": result, "source": "wikipedia"}

    result = search_duckduckgo(query)
    if result:
        return {"found": True, "response": result, "source": "duckduckgo"}

    result = search_google_scrape(query)
    if result:
        return {"found": True, "response": result, "source": "google"}

    return {
        "found": False,
        "response": "К сожалению, не удалось найти информацию по этому запросу. Попробуйте переформулировать вопрос.",
        "source": "none",
    }
