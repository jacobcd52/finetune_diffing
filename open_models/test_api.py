import os
from openai import AzureOpenAI
endpoint = "https://ft-test-sweden.openai.azure.com/"
model_name = "gpt-4o"
deployment = "gpt-4o"
subscription_key = "<API KEY>"
api_version = "2024-12-01-preview"
client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)
response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "I am going to Paris, what should I see?",
        }
    ],
    max_completion_tokens=100000,
    model=deployment
)
print(response)