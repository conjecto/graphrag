# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Create Ollama client instance."""

import logging
from functools import cache

from ollama import AsyncClient

from graphrag.llm import OpenAIConfiguration
from graphrag.llm.ollama.types import OllamaClientTypes

log = logging.getLogger(__name__)

@cache
def create_ollama_client(
    configuration: OpenAIConfiguration
) -> OllamaClientTypes:
    log.info("Creating Ollama client base_url=%s", configuration.api_base)
    return AsyncClient(
        host=configuration.api_base,
        timeout=configuration.request_timeout or 180.0,
    )
