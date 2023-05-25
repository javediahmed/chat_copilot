import openai
import os
import json
import pandas as pd

API_KEY = os.getenv("OPENAI_API_KEY")

class ChatGPT:
    def __init__(self):
        self.api_key = API_KEY
        openai.api_key = self.api_key
        self.model = "text-davinci-002"
        self.messages = [{"role": "system", "content": "You are talking to GPT-3"}]

    def get_user_input(self, prompt):
        user_input = input(prompt).strip()
        if user_input.lower() in ["", "exit", "x"]:
            print("Exiting script.")
            exit(0)
        return user_input

    def process_query_input(self):
        print("Please select the query source:")
        print("1. Enter a query")
        print("2. Load query from a file")
        query_source = self.get_user_input("Enter your choice (1 or 2): ")

        if query_source == "1":
            query = self.get_user_input("Enter your query: ")
        elif query_source == "2":
            filename = self.get_user_input("Enter the file path: ")
            with open(filename, 'r') as file:
                query = file.read()
        else:
            print("Invalid choice. Exiting the script.")
            query = None

        return query

    def process_settings_input(self):
        print("Current Model: ", self.model)
        new_model = self.get_user_input("Enter a new model name or hit enter to keep the current model: ")
        if new_model != '':
            self.model = new_model
        print("Model updated to: ", self.model)

    def chat(self, query):
        self.messages.append({"role": "user", "content": query})
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages,
            max_tokens=100
        )
        return response['choices'][0]['message']['content']

if __name__ == "__main__":
    print("Welcome to the GPT-3 Chat Interface!")
    chatGPT = ChatGPT()

    while True:
        print("\nPlease select an option:")
        print("1. Enter a query")
        print("2. Load query from a file")
        print("3. Settings")
        print("4. Exit")
        user_option = chatGPT.get_user_input("Enter your choice (1 to 4): ")

        if user_option == "1" or user_option == "2":
            query = chatGPT.process_query_input()
            response = chatGPT.chat(query)
            print("Response from GPT-3: ", response)
        elif user_option == "3":
            chatGPT.process_settings_input()
        elif user_option == "4":
            print("Exiting script.")
            break
        else:
            print("Invalid option. Please select again.")
