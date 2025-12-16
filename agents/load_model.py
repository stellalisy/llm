# from .vllm import VllmAgent, VllmBaseAgent

def load_model(model_name, num_gpus=2, **kwargs):
    if model_name.startswith("gpt") and "oss" not in model_name: 
        # in ["gpt-4.1", "gpt-4o", "gpt-4-turbo", "gpt-4o-2024-05-13", "gpt-4-turbo-2024-04-09", "gpt-3.5-turbo", "gpt-3.5-turbo-0301", "gpt-4-0314", "gpt-4-0125-preview", "gpt-4-0613", "gpt-3.5-turbo-0125", "gpt-4o-mini", "gpt-4o-2024-08-06", "gpt-4.1-standard"]:
        from .gpt import AsyncConversationalGPTBaseAgent, ConversationalGPTBaseAgent
        # model = AsyncConversationalGPTBaseAgent({'model': model_name, **kwargs})
        model = ConversationalGPTBaseAgent({'model': model_name, **kwargs})
    elif model_name.startswith("o3") or model_name.startswith("o4") or model_name.startswith("o1"):
        from .gpt import AsyncConversationalGPTReasoningAgent, ConversationalGPTReasoningAgent
        model = ConversationalGPTReasoningAgent({'model': model_name, **kwargs})
    elif model_name.startswith("gemini-"):
        from .gemini import AsyncGeminiAgent
        model = AsyncGeminiAgent({'model': model_name, **kwargs})
    elif model_name.startswith("anthropic") or model_name.startswith("claude"):
        from .claude import AsyncClaudeAgent
        model = AsyncClaudeAgent({'model': model_name, **kwargs})
    elif kwargs.get("api_account", "fair-gpt-4o").startswith("matrix-"):
        from .matrix import AsyncMatrixAgent
        model = AsyncMatrixAgent({'model': model_name, **kwargs})
    elif "oss" in model_name:
        from .huggingface import OssAgent
        model = OssAgent({'model': model_name, **kwargs})
    else:
        from .huggingface import HuggingFaceAgent
        model = HuggingFaceAgent({'model': model_name, **kwargs})
    # elif "meta-llama/" in model_name:
    #     from .llama_vllm import AsyncLlama3Agent
    #     model = AsyncLlama3Agent({'model': model_name, **kwargs})
    # elif model_name in ["mistralai/Mixtral-8x22B-Instruct-v0.1", "mistralai/Mixtral-8x7B-Instruct-v0.1", 'meta-llama/Llama-3-8b-chat-hf',"allenai/OLMo-7B-Instruct",'meta-llama/Llama-3-70b-chat-hf','mistralai/Mistral-7B-Instruct-v0.3','meta-llama/Llama-2-7b-chat-hf']:
    #     from .together_ai import AsyncTogetherAIAgent, AsyncLlama3Agent
    #     model = AsyncTogetherAIAgent({'model': model_name, 'temperature': 0, 'max_tokens': 1024, **kwargs})
    # elif model_name in ["meta-llama/Llama-3-70b-chat-hf-tg"]:
    #     from .together_ai import AsyncTogetherAIAgent, AsyncLlama3Agent
    #     model = AsyncLlama3Agent({'model': model_name, 'temperature': 1.0, 'max_tokens': 1024, **kwargs})
    # elif model_name in ["meta-llama/Llama-2-13b-chat-hf", "mistralai/Mistral-7B-Instruct-v0.2", "HuggingFaceH4/zephyr-7b-beta", "meta-llama/Meta-Llama-3-8B-Instruct"]:
    #     model = VllmAgent(model_name, num_gpus=num_gpus, **kwargs)
    # else:
    #     raise NotImplementedError(f"Model {model_name} not implemented")

    return model