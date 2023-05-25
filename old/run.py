class ChatGPT:
    # existing methods...

    def run(self):
        print("Test GPT interface\n" + str(datetime.today().strftime('%Y-%m-%d')))
        while True:
            print("\nMain Menu:")
            for key, value in MAIN_MENU.items():
                if callable(value):
                    print(f"{key}. {value.__name__.replace('_', ' ').capitalize()}")
                else:
                    print(f"{key}. {value}")
            user_choice = self.prompt_user("Enter your choice: ")
            if user_choice in MAIN_MENU:
                if callable(MAIN_MENU[user_choice]):
                    MAIN_MENU[user_choice]()
                else:
                    break
            else:
                print("Invalid option. Please try again.")

if __name__ == "__main__":
    chat_gpt = ChatGPT()
    MAIN_MENU = {
        "1": chat_gpt.chat,
        "2": chat_gpt.copilot,
        "3": chat_gpt.update_settings,
        "4": chat_gpt.save_results,
        "5": "Exit"
    }
    chat_gpt.run()
