from openai import OpenAI
import base64
import requests
import os
from pdb import set_trace
import argparse
API_KEY = "sk-HHCb4mYKPg0GUcNN4fDc3c34B90648B89d874fFd138f33C3"
BASE_URL = "https://api.kwwai.top/v1"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def get_all_files(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    file_paths.sort()
    return file_paths

def gpt_text_circle():
    # 设置环境变量
    messages = [ {"role": "system", "content": 
                "You are a intelligent assistant."} ]
    client = OpenAI(
        base_url= BASE_URL,
        api_key= API_KEY
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

def gpt_text(report_path):
    # 设置环境变量
    messages = [ {"role": "system", "content": 
                "你是一个金融从业人员，非常善于解读财报，可以洞察到财报很多信息点。."} ]
    client = OpenAI(
        base_url= BASE_URL,
        api_key= API_KEY
    )
    # set_trace()
    
    # message = input("User : ")
        
    message = "下面是一份完整的财报，根据财报回答几个问题：\
                1 企业是否通过技术合作、并购等方式引入外部创新资源， 以提升创新能力。\
                2 年报中是否披露的外部创新项目参与情况及成果转化\
                3 年报中是否提到的激励措施，如对研发人员的激励政策、股权激励等\
                4 企业是否提及专门的创新管理体系或创新部门\
                这几个问题先回答是否，然后简略回答答案的出处, 一定要尊重事实，不要凭空捏造\n\n"

    txt_content = None
    with open(report_path, "r", encoding='utf-8') as f:
        txt_content = f.read()
    message = message + txt_content
    set_trace()
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
    return reply
    # messages.append({"role": "assistant", "content": reply})

def gpt_image(image_path ):
    # OpenAI API Key
    base_url = BASE_URL +"/chat/completions"
    api_key = API_KEY

    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }

    payload = {
    "model": "gpt-4o",
    "messages": [
        {

        "role": "system", 
        "content":[
            {"type" : "text",
             "text" : "你是一个非常厉害的orc软件，可以把图片里的论文内容识别成文字内容，对于结构化的内容，比如图片和表格等，你也可以尽可能用字符来表示，方便后面进一步的文本处理工作。"
            }
             ],
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "把这个图片转化成文本"
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": 3000
    }

    response_raw = requests.post(base_url, headers=headers, json=payload, verify=False)
    
    content = response_raw.json()['choices'][0]['message']['content']

    return content

def save_content(content, file_name):
    with open(file_name, "w") as f:
        f.write(content)
    return
        
def ocr():
    image_path = "test/images"
    content_path = "test/text"
    all_images = get_all_files(image_path)
    all_text = []
    for idx, image_path in enumerate(all_images, start=1):
        # set_trace()
        content = gpt_image(image_path)
        content_name = content_path +"_"+str(idx)+".txt"
        save_content(content, content_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dive in report")
    parser.add_argument('--report', '-i', type=str, help='Path to the report')
    args = parser.parse_args()
    gpt_text(args.report)



    
