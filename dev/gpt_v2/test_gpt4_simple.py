import openai
import json
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

MODELS = {
    "gpt-1.0-turbo": "text-gpt-1-en-12b",
    "gpt-2.0-turbo": "text-gpt-2-en-117b",
    "gpt-3.0-turbo": "text-davinci-002",
    "gpt-3.5-turbo": "text-davinci-003",
    "gpt-4.0-turbo": "text-davinci-004",
    "jurassic-1.0-turbo": "text-jurassic-1-jumbo-en-175b",
    "megatron-turing-nlg-1.0-turbo": "text-megatron-turing-nlg-345m-355b",
    "wudao-2.0-turbo": "text-wudao-2-0-en-1.76T"
}

DEFAULT_SETTINGS = {
    "Model": "gpt-3.0-turbo",
    "Max tokens": 150,
    "Temperature": 0.5,
}

class OpenAIGPTChat:
    def __init__(self):
        self.messages = []
        self.settings = DEFAULT_SETTINGS.copy()

    def add_user_message(self, content):
        self.messages.append({"role": "user", "content": content})

    def add_system_message(self, content):
        self.messages.append({"role": "system", "content": content})

    def generate_response(self):
        model_name = self.settings['Model']
        model_value = MODELS[model_name]
        response = openai.ChatCompletion.create(
            model=model_value,
            messages=self.messages,
            max_tokens=self.settings["Max tokens"],
            temperature=self.settings["Temperature"]
        )
        response_message = response['choices'][0]['message']['content']
        self.add_system_message(response_message)
        return response_message

    def get_conversation(self):
        return self.messages

#Example usage
if __name__ == "__main__":
    chat_model = OpenAIGPTChat()
    chat_model.add_system_message("You are a helpful assistant.")
    chat_model.add_user_message("Who won the world series in 2020?")
    print(chat_model.generate_response())
