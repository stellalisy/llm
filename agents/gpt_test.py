from openai import AzureOpenAI
import yaml
import os

info="""
  api_type: "azure"
  command: "secrets_tool get_group AZ_FAIR_OPENAI1_EASTUS2N3_SAGE"
  secret: "	AZ_FAIR_OPENAI1_EASTUS2N3_SAGE"
  model_name: "gpt-5-mini"
  api_key: "ad288df7a2b649f084f2aff4eeb82fc0"
  SECONDARY_KEY: 63ff1711c5054a7d8f6343a4cb589cfd
  api_base: "https://azure-services-fair-openai1-eastus2n3.azure-api.net"
  api_version: "2025-03-01-preview"
"""

config = yaml.safe_load(info)

client = AzureOpenAI(api_key=config.get("api_key"),
                        azure_endpoint=config.get("api_base"),
                        api_version=config.get("api_version"))

response = client.chat.completions.create(model=config.get("model_name", "gpt-4.1"),
                                                    messages=[{"role": "user", "content": "What is the capital of France?"}],
                                                    # temperature=0.7,
                                                    # max_tokens=4096,
                                                    # top_p=0.95,
                                                    frequency_penalty=0.0,
                                                    presence_penalty=0.0
                                                    )

print(response)
print(response.choices[0].message.content)