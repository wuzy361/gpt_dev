import openai
from openai import OpenAI
import os
from pdb import set_trace

# 设置环境变量
messages = [ {"role": "system", "content": 
              "You are a intelligent assistant."} ]
client = OpenAI(
    base_url="https://api.kwwai.top/v1",
    api_key="sk-HHCb4mYKPg0GUcNN4fDc3c34B90648B89d874fFd138f33C3"
)

# set_trace()
while True:
    message = input("User : ")
    if message:
        messages.append(
            {"role": "user", "content": message},
        )
        chat = client.chat.completions.create(
            model="gpt-4o", messages=messages
        )
        # set_trace()
    reply = chat.choices[0].message.content
    print(f"ChatGPT: {reply}")
    messages.append({"role": "assistant", "content": reply})