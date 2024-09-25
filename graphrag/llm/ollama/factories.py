# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Factory functions for creating OpenAI LLMs."""

import asyncio

from graphrag.llm import LLMCache, LLMLimiter, LLMInvocationFn, ErrorHandlerFn, OnCacheActionFn, EmbeddingLLM, \
    CompletionLLM, OpenAIConfiguration
from graphrag.llm.ollama.ollama_embeddings_llm import OllamaEmbeddingsLLM
from graphrag.llm.ollama.ollama_llm import OllamaLLM
from graphrag.llm.ollama.types import OllamaClientTypes
from graphrag.llm.openai.json_parsing_llm import JsonParsingLLM
from graphrag.llm.openai.openai_history_tracking_llm import OpenAIHistoryTrackingLLM
from graphrag.llm.openai.openai_token_replacing_llm import OpenAITokenReplacingLLM


def create_ollama_llm(
    client: OllamaClientTypes,
    config: OpenAIConfiguration,
    cache: LLMCache | None = None,
    limiter: LLMLimiter | None = None,
    semaphore: asyncio.Semaphore | None = None,
    on_invoke: LLMInvocationFn | None = None,
    on_error: ErrorHandlerFn | None = None,
    on_cache_hit: OnCacheActionFn | None = None,
    on_cache_miss: OnCacheActionFn | None = None,
) -> CompletionLLM:
    """Create an OpenAI chat LLM."""
    operation = "chat"
    result = OllamaLLM(client, config)
    result.on_error(on_error)
    # @todo
    # if limiter is not None or semaphore is not None:
    #     result = _rate_limited(result, config, operation, limiter, semaphore, on_invoke)
    # if cache is not None:
    #     result = _cached(result, config, operation, cache, on_cache_hit, on_cache_miss)
    result = OpenAIHistoryTrackingLLM(result)
    result = OpenAITokenReplacingLLM(result)
    return JsonParsingLLM(result)

def create_ollama_embedding_llm(
    client: OllamaClientTypes,
    config: OpenAIConfiguration,
    cache: LLMCache | None = None,
    limiter: LLMLimiter | None = None,
    semaphore: asyncio.Semaphore | None = None,
    on_invoke: LLMInvocationFn | None = None,
    on_error: ErrorHandlerFn | None = None,
    on_cache_hit: OnCacheActionFn | None = None,
    on_cache_miss: OnCacheActionFn | None = None,
) -> EmbeddingLLM:
    """Create an OpenAI embeddings LLM."""
    operation = "embedding"
    result = OllamaEmbeddingsLLM(client, config)
    result.on_error(on_error)
    # @todo
    # if limiter is not None or semaphore is not None:
    #     result = _rate_limited(result, config, operation, limiter, semaphore, on_invoke)
    # if cache is not None:
    #     result = _cached(result, config, operation, cache, on_cache_hit, on_cache_miss)
    return result
