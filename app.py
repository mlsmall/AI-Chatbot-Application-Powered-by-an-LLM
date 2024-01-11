import os
import io
import base64 
import requests 
requests.adapters.DEFAULT_TIMEOUT = 60

import subprocess

def install_package(package_name):
    subprocess.run(['pip', 'install', package_name])

# Usage:
if __name__ == "__main__":
    install_package("dotenv")

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = os.environ['HF_API_KEY']

# Helper function
import requests, json
from text_generation import Client

#FalcomLM-instruct endpoint on the text_generation library
client = Client(os.environ['HF_API_FALCOM_BASE'], headers={"Authorization": f"Bearer {hf_api_key}"}, timeout=120)

prompt = "Has math been invented or discovered?"
client.generate(prompt, max_new_tokens=256).generated_text

import gradio as gr
def generate(input, slider):
    output = client.generate(input, max_new_tokens=slider).generated_text
    return output

demo = gr.Interface(fn=generate, 
                    inputs=[gr.Textbox(label="Prompt"), 
                            gr.Slider(label="Max new tokens", 
                                      value=20,  
                                      maximum=1024, 
                                      minimum=1)], 
                    outputs=[gr.Textbox(label="Completion")])

gr.close_all()
demo.launch(share=True)#, server_port=int(os.environ['PORT1']))