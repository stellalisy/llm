from google import genai

client = genai.Client(api_key="")

contents = "What is the capital of France?"
model_name = "gemini-2.5-flash"

output = client.models.generate_content(
            model=model_name,
            contents=contents
        )

print(output)