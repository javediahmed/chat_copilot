import openai
import os
import json
from datetime import datetime

# Obtain API key from environment variables
API_KEY = os.getenv("OPENAI_API_KEY")

# Define GPT models
MODELS = {
    "GPT-1": "text-gpt-1-en-12b",
    "GPT-2": "text-gpt-2-en-117b",
    "GPT-3": "text-davinci-002",
    "GPT-3.5": "text-davinci-003",
    "GPT-4": "text-davinci-004",
    "Jurassic-1 Jumbo": "text-jurassic-1-jumbo-en-175b",
    "Megatron-Turing NLG": "text-megatron-turing-nlg-345m-355b",
    "WuDao 2.0": "text-wudao-2-0-en-1.76T"
}

# Default settings
DEFAULT_SETTINGS = {
    "model": MODELS["GPT-3"],
    "max_tokens": 60,
    "temperature": 0.5
}

class ChatGPT:
    """
    This class represents a connection to OpenAI's GPT-3 model.

    It uses the OpenAI Python package to create a chat environment where a user can make queries to the model,
    update settings, and save the history of queries and responses.
    """

    def __init__(self):
        """
        Initializes the GPT-3 chat with API key, default settings, and an empty history.
        """
        openai.api_key = API_KEY
        self.settings = DEFAULT_SETTINGS.copy()
        self.history = []

    def prompt_user(self, message):
        """
        Prompts the user for input with a given message.

        Args:
            message (str): The prompt message.

        Returns:
            str: The user's input.
        """
        return input(message).strip()

    def chat(self, query):
        """
        Makes a query to the GPT-3 model and prints the model's response.

        Args:
            query (str): The user's query.

        Returns:
            str: The model's response.
        """
        response = openai.Completion.create(
            engine=self.settings["model"],
            prompt=query,
            max_tokens=self.settings["max_tokens"],
            temperature=self.settings["temperature"]
        )
        print(response.choices[0].text.strip())
        self.history.append({"query": query, "response": response.choices[0].text.strip()})
        return response.choices[0].text.strip()

    def query_from_file(self, filename):
        """
        Makes a query to the GPT-3 model from a JSON file and prints the model's response.

        Args:
            filename (str): The path to the JSON file.

        Returns:
            str: The model's response.
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                query = data['query']
                return self.chat(query)
        except FileNotFoundError:
            print("File not found. Please try again.")

    def update_settings(self):
        """
        Allows the user to update the model settings.

        The user is shown the current settings and is given the option to update the model, maximum tokens, and temperature.
        """
        print("Current settings:")
        for key, value in self.settings.items():
            print(f"{key}: {value}")

        setting_choice = self.prompt_user("Select the setting to update (m: model, x: max tokens, t: temperature): ")
        if setting_choice == 'm':
            model_choice = self.prompt_user("Select the model (enter model name): ")
            if model_choice in MODELS.keys():
                self.settings["model"] = MODELS[model_choice]
        elif setting_choice == 'x':
            max_tokens_choice = self.prompt_user("Enter the maximum number of tokens: ")
            self.settings["max_tokens"] = int(max_tokens_choice)
        elif setting_choice == 't':
            temperature_choice = self.prompt_user("Enter the temperature (between 0 and 1): ")
            self.settings["temperature"] = float(temperature_choice)

def main():
    chat_gpt = ChatGPT()
    print("Test GPT interface\n" + str(datetime.today().strftime('%Y-%m-%d')))
    while True:
        user_choice = chat_gpt.prompt_user("\nSelect an option:\n1. Chat with model\n2. Enter query file (JSON)\n3. Settings\n4. Exit\n")
        if user_choice == '1':
            query = chat_gpt.prompt_user("Enter your query: ")
            chat_gpt.chat(query)
        elif user_choice == '2':
            filename = chat_gpt.prompt_user("Enter the filename: ")
            chat_gpt.query_from_file(filename)
        elif user_choice == '3':
            chat_gpt.update_settings()
        elif user_choice == '4' or user_choice.lower() == 'exit' or user_choice.lower() == 'x':
            print("Exiting the script.")
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()
