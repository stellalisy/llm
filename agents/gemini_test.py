# from google import genai

# client = genai.Client(api_key="AIzaSyCJas31WH3od8dER02TjPBBQCB9jlsfgQA")

# contents = "What is the capital of France?"
# model_name = "gemini-2.5-flash"

# output = client.models.generate_content(
#             model=model_name,
#             contents=contents
#         )

# print(output)
# print()
# response_text = output.candidates[0].content.parts[0].text
# print(response_text)




from google import genai
client = genai.Client(
        api_key=""
)


response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents="How are you today?"
)
print(response.text)
