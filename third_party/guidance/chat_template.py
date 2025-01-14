from transformers import AutoTokenizer
import base64

tokenizer = AutoTokenizer.from_pretrained("liuhaotian/llava-v1.6-34b")

def encode_image(image_path):
    if image_path.startswith("http"):
        return image_path
    else:
        with open(image_path, "rb") as image_file:
            return f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"
        
url = "longs_peak.jpg"
base64_image = encode_image(url)

chat=[
    {
        "role": "user",
        "content": [
            {
                "type": "text", "text": "Describe following image."
            },
            {
                "type": "image_url",
                "image_url": 
                {
                    "url": base64_image,
                },
            },
        ],
    }
],

template = (
    "{% for message in messages %}"
    "{% if message['role'] != 'system' %}"
    "{{ message['role'].upper() + ': '}}"
    "{% endif %}"
    "{# Render all images first #}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}"
    "{{ '<image>\n' }}"
    "{% endfor %}"
    "{# Render all text next #}"
    "{% if message['role'] != 'assistant' %}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
    "{{ content['text'] + ' '}}"
    "{% endfor %}"
    "{% else %}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
    "{% generation %}"
    "{{ content['text'] + ' '}}"
    "{% endgeneration %}"
    "{% endfor %}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ 'ASSISTANT:' }}"
    "{% endif %}"
)
print("\n")
print(repr(template))
print("\n")

prompt = tokenizer.apply_chat_template(
    chat,
    chat_template=template,
    add_generation_prompt=False,
    tokenize=False
)
print(prompt)