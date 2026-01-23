import os
from openai import OpenAI

client = OpenAI(
    api_key="c27e5d4a-097d-4c3b-9f7f-96f983f49737",
    base_url="https://kspmas.ksyun.com/v1/",
)

# Non-streaming:
print("----- standard request -----")
completion = client.chat.completions.create(
    model="mco-1",  # your model endpoint ID
    messages=[
        {"role": "system", "content": "你是人工智能助手"},
        {"role": "user", "content": "常见的十字花科植物有哪些？"},
    ],
)
print(completion.choices[0].message.content)

# Streaming:
print("----- streaming request -----")
stream = client.chat.completions.create(
    model="mco-1",  # your model endpoint ID
    messages=[
        {"role": "system", "content": "你是人工智能助手"},
        {"role": "user", "content": "常见的十字花科植物有哪些？"},
    ],
    stream=True,
)

for chunk in stream:
    if not chunk.choices:
        continue
    print(chunk.choices[0].delta.content, end="")
print()