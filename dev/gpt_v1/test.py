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
DEFAULT_MODEL = MODELS["GPT-3"]
DEFAULT_MAX_TOKENS = 60
DEFAULT_TEMPERATURE = 0.5

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
        self.settings = {
            "model": DEFAULT_MODEL,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "temperature": DEFAULT_TEMPERATURE
        }
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

        model_choice = self.prompt_user(f"Select the model (Current options are {', '.join(MODELS.keys())}): ")
        if model_choice in MODELS.keys():
            self.settings["model"] = MODELS[model_choice]
        else:
            print("Invalid model choice. Default model will be used.")
        
        max_tokens_choice = self.prompt_user("Enter max tokens (integer, e.g. 60): ")
        try:
            max_tokens_choice = int(max_tokens_choice)
            self.settings["max_tokens"] = max_tokens_choice
        except ValueError:
            print("Invalid choice for max tokens. Default value will be used.")

        temperature_choice = self.prompt_user("Enter temperature (float between 0 and 1, e.g. 0.5): ")
        try:
            temperature_choice = float(temperature_choice)
            self.settings["temperature"] = temperature_choice
        except ValueError:
            print("Invalid choice for temperature. Default value will be used.")

    def save_history(self):
        """
        Saves the history of queries and responses to a JSON file.

        The file is saved as history.json in the current directory.
        """
        with open("history.json", 'w') as f:
            json.dump(self.history, f)
        print("History saved.")

    def process_user_response(self, response):
        """
        Process user response based on the main menu.

        Args:
            response (str): The user's response to the main menu.

        Returns:
            bool: True if the user wants to exit, False otherwise.
        """
        if response == "1":
            query = self.prompt_user("Enter your query: ")
            self.chat(query)
        elif response == "2":
            filename = self.prompt_user("Enter the filename: ")
            self.query_from_file(filename)
        elif response.lower() == "s":
            self.update_settings()
        elif response.lower() == "e" or response == "":
            return True
        else:
            print("Invalid option. Please try again.")
        return False

    def main_menu(self):
        """
        Displays the main menu and prompts the user for their choice.

        The user can choose to make a query, make a query from a file, update settings, or exit.
        """
        print("\nMain Menu:")
        print("1. Make a query")
        print("2. Make a query from a file")
        print("S. Settings")
        print("E. Exit")

        user_choice = self.prompt_user("Enter your choice: ")
        return self.process_user_response(user_choice)


if __name__ == "__main__":
    chat = ChatGPT()

    print(f"\n{'-' * 20}\nTest GPT interface\n{'-' * 20}")
    print("Today's Date:", datetime.now().strftime('%Y-%m-%d'))

    exit = False
    while not exit:
        exit = chat.main_menu()
