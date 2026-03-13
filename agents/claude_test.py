import boto3,json
client = boto3.client(
    "bedrock-runtime",
    region_name="us-west-2",  # or your region
    aws_access_key_id="",
    aws_secret_access_key="",
    aws_session_token="",
)

body = json.dumps({
  "max_tokens": 256,
  "messages": [{"role": "user", "content": "Hello, world"}],
  "anthropic_version": "bedrock-2023-05-31"
})

response = client.invoke_model(body=body, modelId="arn:aws:bedrock:us-west-2:396608793503:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0")

response_body = json.loads(response.get("body").read())
print(response_body.get("content"))