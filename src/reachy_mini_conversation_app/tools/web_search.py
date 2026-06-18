import re
import asyncio
import logging
from typing import Any, Dict

import httpx

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

logger = logging.getLogger(__name__)

_UA = "Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36 Chrome/120.0 Mobile Safari/537.36"
_TIMEOUT = 8.0


class WebSearch(Tool):
    """Search the internet for weather, news, or general information."""

    name = "web_search"
    description = (
        "Search the internet for weather, news, or general information. "
        "Use this when the user asks about current events, weather, or anything "
        "that requires up-to-date information from the web. "
        "Supports weather queries, news lookup, and general web search."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query in Chinese or English",
            },
            "search_type": {
                "type": "string",
                "enum": ["weather", "news", "search"],
                "description": (
                    "Type of search: 'weather' for weather info, "
                    "'news' for latest news, 'search' for general web search"
                ),
            },
            "city": {
                "type": "string",
                "description": "City name for weather queries (optional, defaults to Shenzhen)",
            },
        },
        "required": ["query", "search_type"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        query = kwargs.get("query", "").strip()
        search_type = kwargs.get("search_type", "search")
        city = kwargs.get("city", "").strip() or "Shenzhen"

        if not query:
            return {"error": "query must be a non-empty string"}

        logger.info("Tool call: web_search type=%s query=%s city=%s", search_type, query[:80], city)

        try:
            if search_type == "weather":
                return await asyncio.to_thread(self._search_weather, query, city)
            elif search_type == "news":
                return await asyncio.to_thread(self._search_news, query)
            else:
                return await asyncio.to_thread(self._search_web, query)
        except Exception as e:
            logger.error("web_search error: %s", e)
            return {"error": f"Search failed: {e}"}

    # ── Weather ──────────────────────────────────────────────────

    @staticmethod
    def _search_weather(query: str, city: str) -> Dict[str, Any]:
        headers = {"User-Agent": _UA}

        # Try wttr.in first
        try:
            url = f"https://wttr.in/{city}?format=j1"
            with httpx.Client(timeout=_TIMEOUT, follow_redirects=True) as client:
                r = client.get(url, headers=headers)
            if r.status_code == 200:
                d = r.json()
                c = d["current_condition"][0]
                desc_list = c.get("lang_zh", c.get("weatherDesc", [{}]))
                desc = desc_list[0].get("value", "N/A") if desc_list else "N/A"
                today = d["weather"][0]
                result = {
                    "city": city,
                    "temperature_c": c["temp_C"],
                    "feels_like_c": c["FeelsLikeC"],
                    "humidity": c["humidity"] + "%",
                    "description": desc,
                    "today_high_c": today["maxtempC"],
                    "today_low_c": today["mintempC"],
                }
                if len(d["weather"]) > 1:
                    tmr = d["weather"][1]
                    result["tomorrow_high_c"] = tmr["maxtempC"]
                    result["tomorrow_low_c"] = tmr["mintempC"]
                logger.info("Weather (wttr.in): %s", result)
                return result
        except Exception as e:
            logger.warning("wttr.in failed: %s, falling back to baidu", e)

        # Fallback: Baidu search for weather
        return WebSearch._baidu_search(f"{city}天气")

    # ── News ─────────────────────────────────────────────────────

    @staticmethod
    def _search_news(query: str) -> Dict[str, Any]:
        headers = {"User-Agent": _UA}

        # Try Sina news API
        try:
            url = "https://feed.mix.sina.com.cn/api/roll/get?pageid=153&lid=2509&k=&num=10&page=1"
            with httpx.Client(timeout=_TIMEOUT) as client:
                r = client.get(url, headers=headers)
            if r.status_code == 200:
                data = r.json()
                items = data.get("result", {}).get("data", [])
                headlines = []
                for item in items[:8]:
                    title = item.get("title", "")
                    if title:
                        headlines.append(title)
                if headlines:
                    logger.info("News (sina): %d headlines", len(headlines))
                    return {"source": "sina_news", "headlines": headlines}
        except Exception as e:
            logger.warning("Sina news failed: %s", e)

        # Try Toutiao
        try:
            url = "https://www.toutiao.com/hot-event/hot-board/?origin=toutiao_pc"
            with httpx.Client(timeout=_TIMEOUT, follow_redirects=True) as client:
                r = client.get(url, headers=headers)
            if r.status_code == 200:
                data = r.json()
                headlines = []
                for item in data.get("data", [])[:8]:
                    title = item.get("Title", "")
                    if title:
                        headlines.append(title)
                if headlines:
                    logger.info("News (toutiao): %d headlines", len(headlines))
                    return {"source": "toutiao", "headlines": headlines}
        except Exception as e:
            logger.warning("Toutiao failed: %s", e)

        # Fallback: Baidu search for news
        return WebSearch._baidu_search(query + " 新闻")

    # ── General Web Search ───────────────────────────────────────

    @staticmethod
    def _search_web(query: str) -> Dict[str, Any]:
        return WebSearch._baidu_search(query)

    @staticmethod
    def _baidu_search(query: str) -> Dict[str, Any]:
        headers = {"User-Agent": _UA}
        try:
            url = f"https://m.baidu.com/s?word={query}"
            with httpx.Client(timeout=_TIMEOUT, follow_redirects=True) as client:
                r = client.get(url, headers=headers)
            if r.status_code != 200:
                return {"error": f"Baidu returned status {r.status_code}"}

            html = r.text

            # Extract search result titles using regex
            results = []
            simple_title = re.compile(r'class="c-title-text"[^>]*>(.*?)</span>', re.DOTALL)
            snippet_pattern = re.compile(r'class="c-abstract"[^>]*>(.*?)</div>', re.DOTALL)

            titles = simple_title.findall(html)
            snippets = snippet_pattern.findall(html)

            for i, title in enumerate(titles[:5]):
                clean_title = re.sub(r"<[^>]+>", "", title).strip()
                clean_snippet = ""
                if i < len(snippets):
                    clean_snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip()
                if clean_title:
                    results.append({
                        "title": clean_title,
                        "snippet": clean_snippet[:200] if clean_snippet else "",
                    })

            if results:
                logger.info('Baidu search: %d results for "%s"', len(results), query[:50])
                return {"source": "baidu", "query": query, "results": results}

            # If no structured results, extract text content
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text).strip()
            return {
                "source": "baidu_raw",
                "query": query,
                "content": text[:500],
            }

        except Exception as e:
            logger.error("Baidu search failed: %s", e)
            return {"error": f"Web search failed: {e}"}
