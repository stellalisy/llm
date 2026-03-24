from google import genai

client = genai.Client(api_key="")

contents = "What is the capital of France?"
model_name = "gemini-2.5-flash"

output = client.models.generate_content(
            model=model_name,
            contents=contents
        )

print(output)
print()
response_text = output.candidates[0].content.parts[0].text
print(response_text)