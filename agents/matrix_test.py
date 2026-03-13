import asyncio
from matrix import Cli
from matrix.client import query_llm

metadata = Cli().get_app_metadata(app_name="3B")

# # async call
output = asyncio.run(query_llm.make_request(
  url=metadata["endpoints"]["head"],
  model=metadata["model_name"],
  app_name=metadata["name"],
  data={"messages": [{"role": "user", "content": "Generate anything! Please respond with clean JSON only, without explanations or code blocks."}]},
  guided_decoding = {"json": {"type": "object"}},
))

output = query_llm.batch_requests(
  url=metadata["endpoints"]["head"],
  model=metadata["model_name"],
  app_name=metadata["name"],
  requests=[{"messages": [{"role": "user", "content": "hi"}]}],
)


# batch inference
output = query_llm.batch_requests(
  url=metadata["endpoints"]["head"],
  model=metadata["model_name"],
  app_name=metadata["name"],
  requests=[{"messages": [{"role": "user", "content": "hi"}]}],
)

response_text = output[0]['response']['text'][0]
usage = output[0]['response']['usage']
prompt_tokens = output[0]['response']['usage']['prompt_tokens']
completion_tokens = output[0]['response']['usage']['completion_tokens']

request_timestamp = output[0]['request']['metadata']['request_timestamp'] if 'request_timestamp' in output[0]['request']['metadata'] else None
response_timestamp = output[0]['response']['response_timestamp'] if 'response_timestamp' in output[0]['response'] else None
response_time = response_timestamp - request_timestamp if request_timestamp and response_timestamp else None

output = query_llm.batch_requests(
  url=metadata["endpoints"]["head"],
  model=metadata["model_name"],
  app_name=metadata["name"],
  requests=[{"messages": [{"role": "user", "content": "Generate anything! Please respond with clean JSON only, without explanations or code blocks."}]}],
  guided_decoding = {"json": {"type": "object"}},
)