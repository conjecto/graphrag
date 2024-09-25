from graphrag.llm import OpenAIConfiguration


def get_completion_llm_args(
    parameters: dict | None, configuration: OpenAIConfiguration
) -> dict:
    """Get the arguments for a completion LLM."""
    return {
        **get_completion_cache_args(configuration),
        **(parameters or {}),
    }

def get_completion_cache_args(configuration: OpenAIConfiguration) -> dict:
    """Get the cache arguments for a completion LLM."""
    return {
        "model": configuration.model,
        "options": {
            # @todo https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
            "temperature": configuration.temperature,
            "top_p": configuration.top_p,
            "num_predict": configuration.max_tokens
            # num_ctx ?
        },
        # "frequency_penalty": configuration.frequency_penalty,
        # "presence_penalty": configuration.presence_penalty,
        # "max_tokens": configuration.max_tokens,
        # "n": configuration.n,
    }
