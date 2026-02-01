from openai import AzureOpenAI
import yaml
import os

info="""
  api_type: "azure"
  command: "secrets_tool get_group AZ_FAIR_OPENAI1_WESTUS_SAGE"
  secret: "AZ_FAIR_OPENAI1_WESTUS_SAGE"
  model_name: "gpt-4o"
  api_key: "f5685cb3fcdf414ab6ce103807396265"
  SECONDARY_KEY: 14b61c686ed64239bea38c35b2eaddb0
  api_base: "https://azure-services-fair-openai1-westus.azure-api.net"
  api_version: "2024-06-01"
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