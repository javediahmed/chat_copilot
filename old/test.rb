require 'openai'
require 'json'
require 'date'

OPENAI_API_KEY = ENV['OPENAI_API_KEY']
MODELS = {
  "GPT-3" => "text-davinci-002"
}

DEFAULT_SETTINGS = {
  "Model" => "GPT-3",
  "Query Settings" => {
    "Max Tokens" => 60,
    "Temperature" => 0.5
  },
}

class ChatGPT
  attr_accessor :api_key, :settings, :history

  def initialize
    @api_key = OPENAI_API_KEY
    @settings = DEFAULT_SETTINGS.clone
    @history = []
  end

  def prompt_user(message)
    print "#{message}: "
    gets.strip
  end

  def check_api_key
    while @api_key.nil? || @api_key.empty?
      puts "API Key not found."
      @api_key = prompt_user("Please enter your OpenAI API Key")
    end
    Openai.api_key = @api_key
  end

  def chat
    check_api_key
    model_name = @settings['Model']
    model_value = MODELS[model_name]
    puts "\nChatting with #{model_name} (#{model_value})"
    while true
      query = prompt_user("Enter your query ('x' to exit)")
      break if query.downcase == 'x'
      query_settings = @settings["Query Settings"]
      begin
        response = Openai::Completion.create(
          model: model_value,
          prompt: query,
          max_tokens: query_settings["Max Tokens"],
          temperature: query_settings["Temperature"]
        )
        response_text = response["choices"][0]["text"].strip
        @history.push({"query" => query, "response" => response_text})
        puts response_text
      rescue Openai::ApiError => e
        puts "OpenAI API Error: #{e}"
        @api_key = ""
        check_api_key
      end
    end
  end

  def run
    puts "Test GPT interface\n" + Time.now.strftime('%Y-%m-%d')
    chat
  end
end

if __FILE__ == $0
  chat_gpt = ChatGPT.new
  chat_gpt.run
end
