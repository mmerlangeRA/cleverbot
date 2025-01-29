import asyncio
from typing import List, Dict, Any
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
import asyncio

class SingletonAsyncWebCrawler:
    _instance = None
    _state = "idle"  # Enum-like state: 'idle', 'starting', 'running', 'closing'
    _lock = asyncio.Lock()  # Prevents race conditions when changing states

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = AsyncWebCrawler(*args, **kwargs)
        return cls._instance

    async def start(self) -> None:
        async with self._lock:  # Ensure thread safety for state change
            if self._state == "idle":
                self._state = "starting"
                await self._instance.start()
                self._state = "running"
            else:
                print(f"Crawler cannot be started now, state is {self._state}")

    async def close(self) -> None:
        async with self._lock:  # Ensure thread safety for state change
            if self._state == "running":
                self._state = "closing"
                await self._instance.close()
                self._state = "idle"
            else:
                print(f"Crawler cannot be closed now, state is {self._state}")


async def crawl_url(url: str,crawler:SingletonAsyncWebCrawler=None, crawl_config:CrawlerRunConfig=None, session_id:str="session1")->str:
    """Crawl multiple URLs in parallel with a concurrency limit."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    if crawl_config is None:
        crawl_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            css_selector="article",
            remove_forms=True,
            wait_for_images=False,
            exclude_social_media_links=True,
            exclude_external_images=True
        )

    # Create the crawler instance
    if crawler is None:
        crawler = SingletonAsyncWebCrawler(config=browser_config)
    try:
        await crawler.start()
        result = await crawler.arun(
            url=url,
            config=crawl_config,
            session_id=session_id
        )
        if result.success:
            print(f"Successfully crawled: {url}")
            return result.markdown_v2.raw_markdown
        else:
            print(f"Failed: {url} - Error: {result.error_message}")
    except Exception as e:
        print(f"Error crawling {url}: {e}")

