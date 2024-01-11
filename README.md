{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "c5753a0c",
   "metadata": {},
   "source": [
    "# AI Chatbot Application Powered by an LLM"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a01a3724",
   "metadata": {},
   "source": [
    "We will be using the [Falcon 40B LLM](https://falconllm.tii.ae/falcon.html) model to create a Chatbot application. Falcon 40B is a large language model that has been trained on 40 billion parameters and 1 trillion tokens."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "54acf26e-30a6-475f-aa3e-fda3bfe2342e",
   "metadata": {},
   "source": [
    "#### Setting up the API Key"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a3fc2289-9b93-46d0-9f48-29e1ee9d914c",
   "metadata": {},
   "source": [
    "Load your HuggingFace API key and relevant Python libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "0fa6fa00-6bd1-4839-bcaf-8bae9267ee79",
   "metadata": {
    "height": 217
   },
   "outputs": [],
   "source": [
    "import os\n",
    "import io\n",
    "import base64 \n",
    "import requests \n",
    "requests.adapters.DEFAULT_TIMEOUT = 60\n",
    "\n",
    "from dotenv import load_dotenv, find_dotenv\n",
    "_ = load_dotenv(find_dotenv()) # read local .env file\n",
    "hf_api_key = os.environ['HF_API_KEY']"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2537691a-72c1-4c6a-9800-74dd50a50ff2",
   "metadata": {},
   "source": [
    "The ```text_generation``` is an inference library is used to deal with open source LLM's. It enables you to run to the chatbot application locally and to load both API's: the key that connects us to the Falcom LLM, and the HuggingFace API needed to use the LLM."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "095da8fe-24aa-4dc7-8e08-aa2f949ae21f",
   "metadata": {
    "height": 132
   },
   "outputs": [],
   "source": [
    "# Helper function\n",
    "import requests, json\n",
    "from text_generation import Client\n",
    "\n",
    "# FalcomLM-instruct endpoint on the text_generation library\n",
    "# The client is used to make calls to a text-generation-inference instance\n",
    "client = Client(os.environ['HF_API_FALCOM_BASE'], headers={\"Authorization\": f\"Bearer {hf_api_key}\"}, timeout=120)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bfe6fc97",
   "metadata": {},
   "source": [
    "## Building an app to chat with the Falcon LLM"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "745a3c9b",
   "metadata": {},
   "source": [
    "We create the variable prompt that inlcudes the text we want to feed to the model.  Then we use the client to make the request to the Falcon LLM."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "a7065860-3c0b-490d-9e7c-22e5b79fc004",
   "metadata": {
    "height": 64
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\"\\nThe sky appears blue because of the way that light interacts with the Earth's atmosphere. When light enters the atmosphere, it is scattered by the molecules and particles in the air. The blue light is scattered more than the other colors in the spectrum, which makes it visible to our eyes.\""
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "prompt = \"Why is the sky blue?\"\n",
    "# max_new_tokens is the maxinum number of words in the response\n",
    "client.generate(prompt, max_new_tokens=256).generated_text"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5fbde252",
   "metadata": {},
   "source": [
    "## Creating a Chatbot"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d0530f08",
   "metadata": {},
   "source": [
    "Getting a response from the LLM is great, but it's not really a chatbot. We cannot communicate back and forth with it.\n",
    "To start creating a chatbot, we use Gradio.  First, we use it to generate a user interface that we can use for chatting with our LLM."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "0dcb659e-b71b-46da-b9d2-6ee62498995f",
   "metadata": {
    "height": 302
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Closing server running on port: 7875\n",
      "Closing server running on port: 7860\n",
      "Closing server running on port: 7875\n",
      "Running on local URL:  http://127.0.0.1:7875\n",
      "Running on public URL: https://2c39daa924c37e016f.gradio.live\n",
      "\n",
      "This share link expires in 72 hours. For free permanent hosting and GPU upgrades, run `gradio deploy` from Terminal to deploy to Spaces (https://huggingface.co/spaces)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div><iframe src=\"https://2c39daa924c37e016f.gradio.live\" width=\"100%\" height=\"500\" allow=\"autoplay; camera; microphone; clipboard-read; clipboard-write;\" frameborder=\"0\" allowfullscreen></iframe></div>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": []
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import gradio as gr\n",
    "def generate(input, slider):\n",
    "    output = client.generate(input, max_new_tokens=slider).generated_text\n",
    "    return output\n",
    "\n",
    "demo = gr.Interface(fn=generate, \n",
    "                    inputs=[gr.Textbox(label=\"Prompt\"), \n",
    "                            gr.Slider(label=\"Max new tokens\", \n",
    "                                      value=20,  \n",
    "                                      maximum=1024, \n",
    "                                      minimum=1)], \n",
    "                    outputs=[gr.Textbox(label=\"Completion\")])\n",
    "\n",
    "gr.close_all()\n",
    "demo.launch(share=True)#, server_port=int(os.environ['PORT1']))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b580d369",
   "metadata": {},
   "source": [
    "This interface makes it easy to type our questions and get responses from the model, but it doesn't save our previous questions. So we cannot have a conversation. To fix this, we use another component from Gradio called gr.chatbot()."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8e5f55e2",
   "metadata": {},
   "source": [
    "## Saving the Chat History\n",
    "\n",
    "- `gr.Chatbot()` allows you to save the chat history (between the user and the LLM) as well as display the dialogue in the app.\n",
    "- Define your  to take in a `gr.Chatbot()` object.  \n",
    "  - Within your defined function, append a tuple (or a list) containing the user message and the LLM's response:\n",
    "`chatbot_object.append( (user_message, llm_message) )`\n",
    "\n",
    "- Include the chatbot object in both the inputs and the outputs of the app."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8646d777-c211-4d31-9426-7b5d78b533ae",
   "metadata": {},
   "source": [
    "### Format the prompt with the chat history\n",
    "\n",
    "We create a format function to format the chat prompt to include our message and the chat history. We tell the LLM which messages come from the user (User) and which messages come from the model (Assistant). We then pass the formatted prompt to our API that will generate a response based on both the user and the assistant messages.\n",
    "\n",
    "- You can iterate through the chatbot object with a for loop.\n",
    "- Each item is a tuple containing the user message and the LLM's message.\n",
    "\n",
    "```Python\n",
    "for turn in chat_history:\n",
    "    user_msg, bot_msg = turn\n",
    "    ...\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "321a7017",
   "metadata": {},
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "55bae99d-7a63-4a40-bab7-de7d10b8ab1b",
   "metadata": {
    "height": 489
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Running on local URL:  http://127.0.0.1:7877\n",
      "Running on public URL: https://2ee76c550678000881.gradio.live\n",
      "\n",
      "This share link expires in 72 hours. For free permanent hosting and GPU upgrades, run `gradio deploy` from Terminal to deploy to Spaces (https://huggingface.co/spaces)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div><iframe src=\"https://2ee76c550678000881.gradio.live\" width=\"100%\" height=\"500\" allow=\"autoplay; camera; microphone; clipboard-read; clipboard-write;\" frameborder=\"0\" allowfullscreen></iframe></div>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": []
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "def format_chat_prompt(message, chat_history):\n",
    "    prompt = \"\"\n",
    "    for turn in chat_history:\n",
    "        user_message, bot_message = turn\n",
    "        prompt = f\"{prompt}\\nUser: {user_message}\\nAssistant: {bot_message}\"\n",
    "    prompt = f\"{prompt}\\nUser: {message}\\nAssistant:\"\n",
    "    return prompt\n",
    "\n",
    "def respond(message, chat_history):\n",
    "        formatted_prompt = format_chat_prompt(message, chat_history)\n",
    "        # here we pass the formatted prompt to our client API\n",
    "        bot_message = client.generate(formatted_prompt, max_new_tokens=1024,\n",
    "                                      stop_sequences=[\"\\nUser:\", \"<|endoftext|>\"]).generated_text\n",
    "        chat_history.append((message, bot_message))\n",
    "        return \"\", chat_history\n",
    "\n",
    "with gr.Blocks() as demo:\n",
    "    chatbot = gr.Chatbot(height=240) #just to fit the Jupyter notebook\n",
    "    msg = gr.Textbox(label=\"Prompt\")\n",
    "    btn = gr.Button(\"Submit\")\n",
    "    clear = gr.ClearButton(components=[msg, chatbot], value=\"Clear console\")\n",
    "\n",
    "    btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])\n",
    "    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot]) #Press enter to submit\n",
    "demo.launch(share=True)#, server_port=int(os.environ['PORT3']))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f22b8de8",
   "metadata": {},
   "source": [
    "### Adding other advanced features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9e4fff81-a3d1-4cb8-8d6e-d152ab39065a",
   "metadata": {
    "height": 149
   },
   "outputs": [],
   "source": [
    "def format_chat_prompt(message, chat_history, instruction):\n",
    "    prompt = f\"System:{instruction}\"\n",
    "    for turn in chat_history:\n",
    "        user_message, bot_message = turn\n",
    "        prompt = f\"{prompt}\\nUser: {user_message}\\nAssistant: {bot_message}\"\n",
    "    prompt = f\"{prompt}\\nUser: {message}\\nAssistant:\"\n",
    "    return prompt"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d3ee9bc5-fce7-44b1-af2a-e69bc7c598b6",
   "metadata": {},
   "source": [
    "### Streaming\n",
    "\n",
    "- If your LLM can provide its tokens one at a time in a stream, you can accumulate those tokens in the chatbot object.\n",
    "- The `for` loop in the following function goes through all the tokens that are in the stream and appends them to the most recent conversational turn in the chatbot's message history."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "700eb3bc-b63a-4ccb-94c4-70ec2e54bcda",
   "metadata": {
    "height": 455
   },
   "outputs": [],
   "source": [
    "def respond(message, chat_history, instruction, temperature=0.7):\n",
    "    prompt = format_chat_prompt(message, chat_history, instruction)\n",
    "    chat_history = chat_history + [[message, \"\"]]\n",
    "    stream = client.generate_stream(prompt,\n",
    "                                      max_new_tokens=1024,\n",
    "                                      stop_sequences=[\"\\nUser:\", \"<|endoftext|>\"],\n",
    "                                      temperature=temperature)\n",
    "                                      #stop_sequences to not generate the user answer\n",
    "    acc_text = \"\"\n",
    "    #Streaming the tokens\n",
    "    for idx, response in enumerate(stream):\n",
    "            text_token = response.token.text\n",
    "\n",
    "            if response.details:\n",
    "                return\n",
    "\n",
    "            if idx == 0 and text_token.startswith(\" \"):\n",
    "                text_token = text_token[1:]\n",
    "\n",
    "            acc_text += text_token\n",
    "            last_turn = list(chat_history.pop(-1))\n",
    "            last_turn[-1] += acc_text\n",
    "            chat_history = chat_history + [last_turn]\n",
    "            yield \"\", chat_history\n",
    "            acc_text = \"\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "09873dfd-5b6c-41d6-9479-12e8c8894295",
   "metadata": {
    "height": 268
   },
   "outputs": [],
   "source": [
    "with gr.Blocks() as demo:\n",
    "    chatbot = gr.Chatbot(height=240) #just to fit the notebook\n",
    "    msg = gr.Textbox(label=\"Prompt\")\n",
    "    with gr.Accordion(label=\"Advanced options\",open=False):\n",
    "        system = gr.Textbox(label=\"System message\", lines=2, value=\"A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.\")\n",
    "        temperature = gr.Slider(label=\"temperature\", minimum=0.1, maximum=1, value=0.7, step=0.1)\n",
    "    btn = gr.Button(\"Submit\")\n",
    "    clear = gr.ClearButton(components=[msg, chatbot], value=\"Clear console\")\n",
    "\n",
    "    btn.click(respond, inputs=[msg, chatbot, system], outputs=[msg, chatbot])\n",
    "    msg.submit(respond, inputs=[msg, chatbot, system], outputs=[msg, chatbot]) #Press enter to submit\n",
    "\n",
    "gr.close_all()\n",
    "demo.queue().launch(share=True, server_port=int(os.environ['PORT4']))    "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a4a51a07",
   "metadata": {},
   "source": [
    "Notice, in the cell above, you have used `demo.queue().launch()` instead of `demo.launch()`. \"queue\" helps you to boost up the performance for your demo. You can read [setting up a demo for maximum performance](https://www.gradio.app/guides/setting-up-a-demo-for-maximum-performance) for more details."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "8d9ec80a-39ad-4f58-b79e-4f413c5074c0",
   "metadata": {
    "height": 30
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Closing server running on port: 7875\n",
      "Closing server running on port: 7860\n",
      "Closing server running on port: 7875\n"
     ]
    }
   ],
   "source": [
    "gr.close_all()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "12cf9b3a-4202-4e3a-9c6b-941fa1290ab8",
   "metadata": {
    "height": 30
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
