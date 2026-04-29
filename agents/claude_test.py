import boto3,json
client = boto3.client(
    "bedrock-runtime",
    region_name="us-west-2",  # or your region
    aws_access_key_id="",
    aws_secret_access_key="",
     aws_session_token=""
)


model_name = "anthropic.claude-3-5-sonnet-20241022-v2:0"
body = json.dumps({
  "max_tokens": 256,
  "messages": [{"role": "user", "content": "Hello, world"}],
  "anthropic_version": "bedrock-2023-05-31"
})

response = client.invoke_model(body=body, modelId=f"arn:aws:bedrock:us-west-2:396608793503:inference-profile/us.{model_name}")

response_body = json.loads(response.get("body").read())
print(response_body.get("content"))
