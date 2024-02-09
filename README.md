# Chatbot Application Powered by an LLM

This chatbot is a Generative AI application that will be created using the [Falcon 7B LLM](https://falconllm.tii.ae/falcon.html) model. For this example it will run locally, but it can also run on a server. Falcon 7B is a large language model that has been trained on 7 billion parameters and 1.5 trillion tokens of a massive English web dataset [(RefinedWeb)][https://huggingface.co/datasets/tiiuae/falcon-refinedweb). It was built by the [Technology Innovation Institute](https://www.tii.ae/).

#### Setting up the API Key

To use the Falcon model locally, it is required to connect to it through a HuggingFace API key. So first we load our HuggingFace API key and relevant Python libraries. The key is stored locally in an .env file.


```python
import os
import io
import base64 
import requests 
requests.adapters.DEFAULT_TIMEOUT = 60

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = os.environ['HF_API_KEY']
```

The ```text_generation``` library is an inference library that offers a convenient way of interfacing with open source LLM's. It enables you to run to the chatbot application locally and to load both API's. One for connectiong us to the Falcom LLM, and the second one is HuggingFace API key needed to use the LLM.


```python
# Helper function
import requests, json
from text_generation import Client

# FalcomLM-instruct endpoint on the text_generation library
# The client is used to make calls to a text-generation-inference instance
client = Client(os.environ['HF_API_FALCOM_BASE'], headers={"Authorization": f"Bearer {hf_api_key}"}, timeout=120)
```

### Creating a Prompt to Chat with the Falcon LLM

We create the variable prompt that inlcudes the text we want to feed to the model.  Then we use the client to make the request to the Falcon LLM.


```python
prompt = "Why is the sky blue?"
# max_new_tokens is the maxinum number of words in the response
client.generate(prompt, max_new_tokens=256).generated_text
```

    "The sky appears blue because of the way that light interacts with the Earth's atmosphere. When light enters the atmosphere, it is scattered by the molecules and particles in the air. The blue light is scattered more than the other colors in the spectrum, which makes it visible to our eyes."



## Creating a Chatbot Application
Getting a response from the LLM is great, but we cannot communicate back and forth with it.  It's not really a chatbot unless we can follow up with questions and it understands the context of our conversation.
To create a real chatbot, we will use Gradio.

### Creating the User Interface with Gradio
First, Gradio used to generate a user interface that we can use for chatting with our LLM. [Gradio](https://www.gradio.app/) is an open-source Python package that can be used to quickly build a web application for a machine learning model or API. 

```python
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
demo.launch() # To share the chatbot publicly use the attribute `share=True`. You can use `server_port=int(os.environ['PORT1'])` to specify your own port.
```
Now we have a chatbot interface where we can type a prompt and get a response. This is what it looks like.

<img src="https://github.com/mlsmall/AI-Chatbot-Application-Powered-by-an-LLM/blob/main/first prompt.png" width="1012" />

This interface makes it easy to type our questions and get responses from the model, but it doesn't save our previous questions. So we cannot have a conversation. To fix this, we use another component from Gradio called gr.chatbot().

### Saving the Chat History

- `gr.Chatbot()` allows us to save the chat history (between the user and the LLM) as well as display the dialogue in the application.
- Define a function to take in a `gr.Chatbot()` object.
  - Within the defined function, we append to the chat history a tuple (or a list) containing the user message and the LLM's response: `chat_history.append((message, bot_message))`.
- Include the chatbot object in both the inputs and the outputs of the app.

#### Format the prompt to save the chat history

We create a format function to format the chat prompt to include our message and the chat history. We tell the LLM which messages come from the user (User) and which messages come from the model (Assistant). We then pass the formatted prompt to our API that will generate a response based on both the user and the assistant messages.

- You can iterate through the chatbot object with a for loop.
- Each item is a tuple containing the user message and the LLM's message.

The idea is to create a function like this:
```Python
for turn in chat_history:
    user_message, bot_message = turn
    ...
```
```python
# We format the prompt to include the chat history and we tell it which messages come from the user and which messages come from itself (Assistant).
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
    chatbot = gr.Chatbot(height=240) # height is for the he size of the chatbot interface
    msg = gr.Textbox(label="Prompt") # This is the prompt box where you write your message
    btn = gr.Button("Submit") # The actual button that you click to submit your prompt
    clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")

    btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot]) #Press enter to submit
demo.launch(share=True)#, server_port=int(os.environ['PORT3']))
```
<img src="https://github.com/mlsmall/AI-Chatbot-Application-Powered-by-an-LLM/blob/main/chatbot%20window.png" width="1012" />  

## Advanced Gradio Features
### Creating a system message
You can create a system message that will you give your chatbot assistance a personality.

```python
def format_chat_prompt(message, chat_history, instruction):
    prompt = f"System:{instruction}"
    for turn in chat_history:
        user_message, bot_message = turn
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    return prompt
```

### Streaming

- If your LLM can provide its tokens one at a time in a stream, you can accumulate those tokens in the chatbot object.
- The `for` loop in the following function goes through all the tokens that are in the stream and appends them to the most recent conversational turn in the chatbot's message history.


```python
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


```python
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

demo.launch()
```
<img src="https://github.com/mlsmall/AI-Chatbot-Application-Powered-by-an-LLM/blob/main/advanced%20window%201.png" width="1012" />
<img src="https://github.com/mlsmall/AI-Chatbot-Application-Powered-by-an-LLM/blob/main/advanced%20window%202.png" width="1012" />

And to close all your gradio interfaces:


```python
gr.close_all()
```
