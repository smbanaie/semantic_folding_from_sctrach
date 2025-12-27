"""
OpenRouter API client for LLM interactions.

Provides async HTTP client for making requests to OpenRouter API
with error handling, retries, and token usage logging.
"""

import asyncio
import logging
from typing import Optional
import aiohttp
from aiohttp import ClientError, ClientTimeout

logger = logging.getLogger(__name__)


class OpenRouterClient:
    """
    Async client for OpenRouter API.

    Handles LLM requests with retry logic, error handling,
    and token usage tracking.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: int = 120,
        max_retries: int = 3,
    ):
        """
        Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key
            model: Model identifier (e.g., "meta-llama/llama-3.1-70b-instruct:free")
            base_url: OpenRouter API base URL
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> str:
        """
        Generate text using OpenRouter API.

        Args:
            prompt: User prompt text
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response

        Raises:
            RuntimeError: If all retry attempts fail
            ValueError: If API returns an error response
        """
        if not self.session:
            self.session = aiohttp.ClientSession(timeout=self.timeout)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if response_format:
            payload["response_format"] = response_format

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/knowledge-graph-builder",
            "X-Title": "Knowledge Graph Builder",
        }

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                async with self.session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        result = data["choices"][0]["message"]["content"]

                        # Log token usage if available
                        if "usage" in data:
                            usage = data["usage"]
                            logger.info(
                                f"Token usage - Model: {self.model}, "
                                f"Prompt: {usage.get('prompt_tokens', 0)}, "
                                f"Completion: {usage.get('completion_tokens', 0)}, "
                                f"Total: {usage.get('total_tokens', 0)}"
                            )

                        return result

                    elif response.status == 429:
                        # Rate limit - wait and retry
                        retry_after = int(response.headers.get("Retry-After", 60))
                        logger.warning(
                            f"Rate limited. Waiting {retry_after}s before retry {attempt}/{self.max_retries}"
                        )
                        if attempt < self.max_retries:
                            await asyncio.sleep(retry_after)
                            continue
                        else:
                            error_data = await response.json()
                            raise ValueError(
                                f"Rate limit exceeded: {error_data.get('error', {}).get('message', 'Unknown error')}"
                            )

                    else:
                        # Other HTTP errors
                        error_data = await response.json()
                        error_msg = error_data.get("error", {}).get(
                            "message", f"HTTP {response.status}"
                        )
                        logger.error(f"API error (attempt {attempt}/{self.max_retries}): {error_msg}")

                        if response.status >= 500 and attempt < self.max_retries:
                            # Server error - retry
                            wait_time = 2 ** attempt  # Exponential backoff
                            logger.info(f"Retrying after {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            raise ValueError(f"API error: {error_msg}")

            except ClientError as e:
                last_error = e
                logger.warning(
                    f"Network error (attempt {attempt}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(f"Failed after {self.max_retries} attempts: {e}") from e

            except asyncio.TimeoutError:
                last_error = "Request timeout"
                logger.warning(
                    f"Timeout (attempt {attempt}/{self.max_retries})"
                )
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(
                        f"Request timed out after {self.max_retries} attempts"
                    )

        # If we get here, all retries failed
        raise RuntimeError(
            f"Failed to generate response after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )

    async def close(self):
        """Close the HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None

