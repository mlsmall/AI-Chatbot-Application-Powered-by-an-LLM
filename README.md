---
jupyter:
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.11.0
  nbformat: 4
  nbformat_minor: 5
---

::: {#c5753a0c .cell .markdown}
# AI Chatbot Application Powered by an LLM
:::

::: {#a01a3724 .cell .markdown}
We will be using the [Falcon 40B
LLM](https://falconllm.tii.ae/falcon.html) model to create a Chatbot
application. Falcon 40B is a large language model that has been trained
on 40 billion parameters and 1 trillion tokens.
:::

::: {#54acf26e-30a6-475f-aa3e-fda3bfe2342e .cell .markdown}
#### Setting up the API Key
:::

::: {#a3fc2289-9b93-46d0-9f48-29e1ee9d914c .cell .markdown}
Load your HuggingFace API key and relevant Python libraries
:::

::: {#0fa6fa00-6bd1-4839-bcaf-8bae9267ee79 .cell .code execution_count="33" height="217"}
``` python
import os
import io
import base64 
import requests 
requests.adapters.DEFAULT_TIMEOUT = 60

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = os.environ['HF_API_KEY']
```
:::

::: {#2537691a-72c1-4c6a-9800-74dd50a50ff2 .cell .markdown}
The `text_generation` is an inference library is used to deal with open
source LLM\'s. It enables you to run to the chatbot application locally
and to load both API\'s: the key that connects us to the Falcom LLM, and
the HuggingFace API needed to use the LLM.
:::

::: {#095da8fe-24aa-4dc7-8e08-aa2f949ae21f .cell .code execution_count="34" height="132"}
``` python
# Helper function
import requests, json
from text_generation import Client

# FalcomLM-instruct endpoint on the text_generation library
# The client is used to make calls to a text-generation-inference instance
client = Client(os.environ['HF_API_FALCOM_BASE'], headers={"Authorization": f"Bearer {hf_api_key}"}, timeout=120)
```
:::

::: {#bfe6fc97 .cell .markdown}
## Building an app to chat with the Falcon LLM
:::

::: {#745a3c9b .cell .markdown}
We create the variable prompt that inlcudes the text we want to feed to
the model. Then we use the client to make the request to the Falcon LLM.
:::

::: {#a7065860-3c0b-490d-9e7c-22e5b79fc004 .cell .code execution_count="35" height="64"}
``` python
prompt = "Why is the sky blue?"
# max_new_tokens is the maxinum number of words in the response
client.generate(prompt, max_new_tokens=256).generated_text
```

::: {.output .execute_result execution_count="35"}
    "\nThe sky appears blue because of the way that light interacts with the Earth's atmosphere. When light enters the atmosphere, it is scattered by the molecules and particles in the air. The blue light is scattered more than the other colors in the spectrum, which makes it visible to our eyes."
:::
:::

::: {#5fbde252 .cell .markdown}
## Creating a Chatbot
:::

::: {#d0530f08 .cell .markdown}
Getting a response from the LLM is great, but it\'s not really a
chatbot. We cannot communicate back and forth with it. To start creating
a chatbot, we use Gradio. First, we use it to generate a user interface
that we can use for chatting with our LLM.
:::

::: {#0dcb659e-b71b-46da-b9d2-6ee62498995f .cell .code execution_count="36" height="302"}
``` python
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
```

::: {.output .stream .stdout}
    Closing server running on port: 7875
    Closing server running on port: 7860
    Closing server running on port: 7875
    Running on local URL:  http://127.0.0.1:7875
    Running on public URL: https://2c39daa924c37e016f.gradio.live

    This share link expires in 72 hours. For free permanent hosting and GPU upgrades, run `gradio deploy` from Terminal to deploy to Spaces (https://huggingface.co/spaces)
:::

::: {.output .display_data}
```{=html}
<div><iframe src="https://2c39daa924c37e016f.gradio.live" width="100%" height="500" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>
```
:::

::: {.output .execute_result execution_count="36"}
:::
:::

::: {#b580d369 .cell .markdown}
This interface makes it easy to type our questions and get responses
from the model, but it doesn\'t save our previous questions. So we
cannot have a conversation. To fix this, we use another component from
Gradio called gr.chatbot().
:::

::: {#8e5f55e2 .cell .markdown}
## Saving the Chat History

-   `gr.Chatbot()` allows you to save the chat history (between the user
    and the LLM) as well as display the dialogue in the app.
-   Define your to take in a `gr.Chatbot()` object.
    -   Within your defined function, append a tuple (or a list)
        containing the user message and the LLM\'s response:
        `chatbot_object.append( (user_message, llm_message) )`
-   Include the chatbot object in both the inputs and the outputs of the
    app.
:::

::: {#8646d777-c211-4d31-9426-7b5d78b533ae .cell .markdown}
### Format the prompt with the chat history

We create a format function to format the chat prompt to include our
message and the chat history. We tell the LLM which messages come from
the user (User) and which messages come from the model (Assistant). We
then pass the formatted prompt to our API that will generate a response
based on both the user and the assistant messages.

-   You can iterate through the chatbot object with a for loop.
-   Each item is a tuple containing the user message and the LLM\'s
    message.

``` python
for turn in chat_history:
    user_msg, bot_msg = turn
    ...
```
:::

::: {#321a7017 .cell .markdown}
:::

::: {#55bae99d-7a63-4a40-bab7-de7d10b8ab1b .cell .code execution_count="38" height="489"}
``` python
def format_chat_prompt(message, chat_history):
    prompt = ""
    for turn in chat_history:
        user_message, bot_message = turn
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    return prompt

def respond(message, chat_history):
        formatted_prompt = format_chat_prompt(message, chat_history)
        # here we pass the formatted prompt to our client API
        bot_message = client.generate(formatted_prompt, max_new_tokens=1024,
                                      stop_sequences=["\nUser:", "<|endoftext|>"]).generated_text
        chat_history.append((message, bot_message))
        return "", chat_history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=240) #just to fit the Jupyter notebook
    msg = gr.Textbox(label="Prompt")
    btn = gr.Button("Submit")
    clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")

    btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot]) #Press enter to submit
demo.launch(share=True)#, server_port=int(os.environ['PORT3']))
```

::: {.output .stream .stdout}
    Running on local URL:  http://127.0.0.1:7877
    Running on public URL: https://2ee76c550678000881.gradio.live

    This share link expires in 72 hours. For free permanent hosting and GPU upgrades, run `gradio deploy` from Terminal to deploy to Spaces (https://huggingface.co/spaces)
:::

::: {.output .display_data}
```{=html}
<div><iframe src="https://2ee76c550678000881.gradio.live" width="100%" height="500" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>
```
:::

::: {.output .execute_result execution_count="38"}
:::
:::

::: {#f22b8de8 .cell .markdown}
### Adding other advanced features
:::

::: {#9e4fff81-a3d1-4cb8-8d6e-d152ab39065a .cell .code height="149"}
``` python
def format_chat_prompt(message, chat_history, instruction):
    prompt = f"System:{instruction}"
    for turn in chat_history:
        user_message, bot_message = turn
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    return prompt
```
:::

::: {#d3ee9bc5-fce7-44b1-af2a-e69bc7c598b6 .cell .markdown}
### Streaming

-   If your LLM can provide its tokens one at a time in a stream, you
    can accumulate those tokens in the chatbot object.
-   The `for` loop in the following function goes through all the tokens
    that are in the stream and appends them to the most recent
    conversational turn in the chatbot\'s message history.
:::

::: {#700eb3bc-b63a-4ccb-94c4-70ec2e54bcda .cell .code execution_count="27" height="455"}
``` python
def respond(message, chat_history, instruction, temperature=0.7):
    prompt = format_chat_prompt(message, chat_history, instruction)
    chat_history = chat_history + [[message, ""]]
    stream = client.generate_stream(prompt,
                                      max_new_tokens=1024,
                                      stop_sequences=["\nUser:", "<|endoftext|>"],
                                      temperature=temperature)
                                      #stop_sequences to not generate the user answer
    acc_text = ""
    #Streaming the tokens
    for idx, response in enumerate(stream):
            text_token = response.token.text

            if response.details:
                return

            if idx == 0 and text_token.startswith(" "):
                text_token = text_token[1:]

            acc_text += text_token
            last_turn = list(chat_history.pop(-1))
            last_turn[-1] += acc_text
            chat_history = chat_history + [last_turn]
            yield "", chat_history
            acc_text = ""
```
:::

::: {#09873dfd-5b6c-41d6-9479-12e8c8894295 .cell .code height="268"}
``` python
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=240) #just to fit the notebook
    msg = gr.Textbox(label="Prompt")
    with gr.Accordion(label="Advanced options",open=False):
        system = gr.Textbox(label="System message", lines=2, value="A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.")
        temperature = gr.Slider(label="temperature", minimum=0.1, maximum=1, value=0.7, step=0.1)
    btn = gr.Button("Submit")
    clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")

    btn.click(respond, inputs=[msg, chatbot, system], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot, system], outputs=[msg, chatbot]) #Press enter to submit

gr.close_all()
demo.queue().launch(share=True, server_port=int(os.environ['PORT4']))    
```
:::

::: {#a4a51a07 .cell .markdown}
Notice, in the cell above, you have used `demo.queue().launch()` instead
of `demo.launch()`. \"queue\" helps you to boost up the performance for
your demo. You can read [setting up a demo for maximum
performance](https://www.gradio.app/guides/setting-up-a-demo-for-maximum-performance)
for more details.
:::

::: {#8d9ec80a-39ad-4f58-b79e-4f413c5074c0 .cell .code execution_count="32" height="30"}
``` python
gr.close_all()
```

::: {.output .stream .stdout}
    Closing server running on port: 7875
    Closing server running on port: 7860
    Closing server running on port: 7875
:::
:::

::: {#12cf9b3a-4202-4e3a-9c6b-941fa1290ab8 .cell .code height="30"}
``` python
```
:::
