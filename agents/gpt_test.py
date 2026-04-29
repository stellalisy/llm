from openai import AzureOpenAI
import yaml
import os

info="""
  api_type: "azure"
  command: "secrets_tool get_group AZ_FAIR_OPENAI2_EASTUS2N4_SAGE"
  secret: "AZ_FAIR_OPENAI2_EASTUS2N4_SAGE"
  model_name: "gpt-4.1"
  api_base: "https://azure-services-fair-openai2-eastus2n4.azure-api.net"
  api_key: ""
  secondary_api_key: ""
  model_version: "2025-04-14"
  api_version: "2025-03-01-preview"
"""


config = yaml.safe_load(info)

client = AzureOpenAI(api_key=config.get("api_key"),
                        azure_endpoint=config.get("api_base"),
                        api_version=config.get("api_version"))

response = client.chat.completions.create(model=config.get("model_name", config.get("model_name")),
                                                    messages=[{"role": "user", "content": "What is the capital of France?"}],
                                                    # temperature=0.7,
                                                    # max_tokens=4096,
                                                    # top_p=0.95,
                                                    frequency_penalty=0.0,
                                                    presence_penalty=0.0
                                                    )

print(response)
print()
print(response.choices[0].message.content)
