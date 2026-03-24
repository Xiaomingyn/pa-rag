
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Your DeepSeek API key (sk-...)
api_key=os.getenv("DEEPSEEK_API_KEY")

# Initialize for DeepSeek reasoning model
llm = ChatOpenAI(
    model="deepseek-reasoner",  # Or "deepseek-chat" for non-reasoning
    api_key=api_key,
    base_url="https://api.deepseek.com",  # Key: DeepSeek endpoint
    temperature=0.0  # For reasoning, low temp
)

# Simple invoke
messages = [HumanMessage(content="Solve: If a bat and ball cost $1.10, and bat is $1 more than ball, ball price? Explain step-by-step.")]
response = llm.invoke(messages)
print(response.content)  # Prints answer + reasoning chain

#import requests 
#from openai import OpenAI
# 
#api_key=os.getenv("DEEPSEEK_API_KEY")
#
#url = "https://api.deepseek.com/chat/completions"
#headers = {
#    "Content-Type": "application/json",
#    "Authorization": f"Bearer {api_key}"
#}
#data = {
#    "model": "deepseek-reasoner",  # Reasoning model (R1-like)
#    "messages": [
#        {"role": "user", "content": "Solve: A bat and ball cost $1.10 total. Bat costs $1 more than ball. How much is the ball?"}
#    ],
#    "stream": False
#}
#
#response = requests.post(url, headers=headers, json=data)
#if response.status_code == 200:
#    result = response.json()
#    print(result['choices'][0]['message']['content'])  # Final answer
#    # For reasoning: print(result['choices'][0]['message'].get('reasoning_content', ''))
#else:
#    print("Error:", response.status_code, response.text)
