import openai
import os
import json
from datetime import datetime

API_KEY = os.getenv("OPENAI_API_KEY")

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

MAIN_MENU = {
    "1": "Chat with model",
    "2": "Submit JSON query",
    "3": "Copilot",
    "4": "Settings",
    "5": "Save/export results",
    "6": "Exit"
}

DEFAULT_SETTINGS = {
    "model": MODELS["GPT-3"],
    "max_tokens": 60,
    "temperature": 0.5,
    "role": "user",
    "menu": MAIN_MENU
}


class ChatGPT:
    def __init__(self):
        openai.api_key = API_KEY
        self.settings = DEFAULT_SETTINGS.copy()
        self.history = []

    def prompt_user(self, message):
        return input(message).strip()

    def chat(self, query):
        response = openai.Completion.create(
            engine=self.settings["model"],
            prompt=query,
            max_tokens=self.settings["max_tokens"],
            temperature=self.settings["temperature"]
        )
        print(response.choices[0].text.strip())
        self.history.append({"query": query, "response": response.choices[0].text.strip()})
        return response.choices[0].text.strip()

    def query_from_json(self):
        json_query = self.prompt_user("Enter the JSON query: ")
        try:
            data = json.loads(json_query)
            query = data['query']
            self.chat(query)
        except (json.JSONDecodeError, KeyError):
            print("Invalid JSON query. Please try again.")

    def copilot(self):
        query = self.prompt_user("Enter your Copilot query: ")
        response = self.chat(query)
        self.history[-1]["copilot_response"] = response

    def update_settings(self):
        print("Current settings:")
        for key, value in self.settings.items():
            if key != "menu":
                print(f"{key}: {value}")

    def main_menu(self):
        print("\nSelect an option:")
        for key, value in self.settings["menu"].items():
            print(f"{key}. {value}")

    def process_user_input(self, user_choice):
        if user_choice in self.settings["menu"]:
            option_name = self.settings["menu"][user_choice]
            if option_name == "Chat with model":
                query = self.prompt_user("Enter your query: ")
                self.chat(query)
            elif option_name == "Submit JSON query":
                self.query_from_json()
            elif option_name == "Copilot":
                self.copilot()
            elif option_name == "Settings":
                self.update_settings()
            elif option_name == "Save/export results":
                self.save_results()
            elif option_name == "Exit":
                print("Exiting the script.")
                return False
        else:
            print("Invalid option. Please try again.")

        return True

    def save_results(self):
        filename = self.prompt_user("Enter the filename to save the results: ")
        with open(filename, 'w') as f:
            json.dump(self.history, f)
        print(f"Results saved to {filename}")


def main():
    chat_gpt = ChatGPT()
    print("Test GPT interface\n" + str(datetime.today().strftime('%Y-%m-%d')))
    while True:
        chat_gpt.main_menu()
        user_choice = chat_gpt.prompt_user("Enter your choice: ")
        if not chat_gpt.process_user_input(user_choice):
            break


if __name__ == "__main__":
    main()
