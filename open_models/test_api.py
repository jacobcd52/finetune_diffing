endpoint = "https://ai-anna4195ai013245888589.cognitiveservices.azure.com/"
model_name = "gpt-4o-mini-def"
deployment = "gpt-4o-mini-def"
subscription_key = "Bkjog7f6jPb3BtpxeWsze4lFep44e4juarIXkYDK5qA2T2pYH0s6JQQJ99BBACYeBjFXJ3w3AAAAACOGQynB"
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