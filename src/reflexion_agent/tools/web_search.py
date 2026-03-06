import asyncio
import json
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from reflexion_agent.observability import get_logger
from reflexion_agent.tools.base import (
    BaseTool, 
    ToolResult, 
    ToolValidationError,
    AsyncBaseTool
)

logger = get_logger(__name__)


class SearchResult:
    """Structured search result."""
    
    def __init__(self, title: str, link: str, snippet: str):
        self.title = title
        self.link = link
        self.snippet = snippet
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "title": self.title,
            "link": self.link,
            "snippet": self.snippet
        }
    
    def __str__(self) -> str:
        return f"{self.title}\n{self.link}\n{self.snippet}\n"


class WebSearchTool(AsyncBaseTool):
    """Web search tool using various search APIs."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        search_engine: str = "duckduckgo",  # or "google","duckduckgo," "bing", "tavily"
        max_results: int = 5,
        timeout: float = 10.0,
    ):
        """
        Initialize web search tool.
        
        Args:
            api_key: API key for search service (if required)
            search_engine: Search engine to use
            max_results: Maximum number of results to return
            timeout: Request timeout in seconds
        """
        super().__init__()
        self.api_key = api_key
        self.search_engine = search_engine.lower()
        self.max_results = max_results
        self.timeout = timeout
        self.client: Optional[httpx.AsyncClient] = None
    
    @property
    def name(self) -> str:
        return "web_search"
    
    @property
    def description(self) -> str:
        return """Search the web for current information. 
Use this for real-time data, recent events, or facts you're unsure about.
Returns titles, links, and snippets from search results."""
    
    async def initialize(self):
        """Initialize HTTP client."""
        if not self.client:
            self.client = httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                limits=httpx.Limits(max_keepalive_connections=5)
            )
            logger.debug("web_search_client_initialized")
    
    async def cleanup(self):
        """Cleanup HTTP client."""
        if self.client:
            await self.client.aclose()
            self.client = None
            logger.debug("web_search_client_closed")
    
    def validate_input(self, **kwargs) -> None:
        """Validate search input."""
        query = kwargs.get('query', '')
        
        if not isinstance(query, str):
            raise ToolValidationError("Query must be a string")
        
        if not query.strip():
            raise ToolValidationError("Query cannot be empty")
        
        if len(query) > 500:
            raise ToolValidationError("Query too long (max 500 characters)")
    
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(min=1, max=5),
    )
    async def _search_duckduckgo(self, query: str) -> List[SearchResult]:
        """Search using DuckDuckGo (no API key required)."""
        if not self.client:
            await self.initialize()
        
        # DuckDuckGo HTML scraping (for demo - use official API in production)
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; ReflexionAgent/1.0; +https://github.com/yourusername/reflexion-agent)'
        }
        
        response = await self.client.get(url, headers=headers)
        response.raise_for_status()
        
        # Simple parsing (in production, use proper HTML parsing)
        import re
        results = []
        
        # Extract result blocks
        result_blocks = re.findall(
            r'<a rel="nofollow" class="result__a" href="([^"]+)">([^<]+)</a>.*?<a class="result__snippet"[^>]*>(.*?)</a>',
            response.text,
            re.DOTALL
        )
        
        for link, title, snippet in result_blocks[:self.max_results]:
            # Clean up HTML entities
            snippet = re.sub(r'<[^>]+>', '', snippet)
            snippet = re.sub(r'&[a-z]+;', ' ', snippet)
            
            results.append(SearchResult(
                title=title.strip(),
                link=link,
                snippet=snippet.strip()
            ))
        
        return results
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=10),
    )
    async def _search_tavily(self, query: str) -> List[SearchResult]:
        """Search using Tavily API (requires API key)."""
        if not self.api_key:
            raise ToolValidationError("Tavily API key required")
        
        if not self.client:
            await self.initialize()
        
        url = "https://api.tavily.com/search"
        
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": self.max_results,
            "search_depth": "basic"
        }
        
        response = await self.client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get('results', []):
            results.append(SearchResult(
                title=item.get('title', ''),
                link=item.get('url', ''),
                snippet=item.get('content', '')
            ))
        
        return results
    
    async def _search_google(self, query: str) -> List[SearchResult]:
        """Search using Google Custom Search API."""
        if not self.api_key:
            raise ToolValidationError("Google API key required")
        
        # Implementation for Google Custom Search
        # Requires: api_key and search_engine_id
        raise NotImplementedError("Google search not yet implemented")
    
    async def _search_bing(self, query: str) -> List[SearchResult]:
        """Search using Bing Search API."""
        if not self.api_key:
            raise ToolValidationError("Bing API key required")
        
        # Implementation for Bing Search
        raise NotImplementedError("Bing search not yet implemented")
    
    async def _execute(self, query: str = "", **kwargs) -> str:
        """
        Execute web search.
        
        Args:
            query: Search query
            max_results: Optional override for max results
            
        Returns:
            Formatted search results
        """
        max_results = kwargs.get('max_results', self.max_results)
        
        # Select search method
        if self.search_engine == "duckduckgo":
            results = await self._search_duckduckgo(query)
        elif self.search_engine == "tavily":
            results = await self._search_tavily(query)
        elif self.search_engine == "google":
            results = await self._search_google(query)
        elif self.search_engine == "bing":
            results = await self._search_bing(query)
        else:
            raise ToolValidationError(f"Unknown search engine: {self.search_engine}")
        
        # Format results
        if not results:
            return "No search results found."
        
        output_lines = [f"Search results for: {query}\n"]
        
        for i, result in enumerate(results[:max_results], 1):
            output_lines.append(f"{i}. {result.title}")
            output_lines.append(f"   URL: {result.link}")
            output_lines.append(f"   {result.snippet}")
            output_lines.append("")
        
        logger.info(
            "web_search_completed",
            query=query,
            num_results=len(results),
            engine=self.search_engine
        )
        
        return "\n".join(output_lines)
    
    def parameters_schema(self) -> Dict[str, Any]:
        """Get JSON schema for web search parameters."""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                    "examples": ["current weather in London", "latest AI news"]
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "minimum": 1,
                    "maximum": 20,
                    "default": self.max_results
                }
            },
            "required": ["query"]
        }