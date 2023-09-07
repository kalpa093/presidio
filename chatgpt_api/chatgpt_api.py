import openai

api_key = "api"
openai.api_key = api_key

with open("input.txt", "r") as file:
    user_message = file.read()

# Request의 자세한 변수조정은 https://platform.openai.com/docs/api-reference/making-requests 참고.
response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a PII anonymizer."},
        {"role": "user", "content": user_message}
    ]
)

message = response['choices'][0]['message']['content']
print(message)

with open("output.txt", "w") as file:
    file.write(model_response)