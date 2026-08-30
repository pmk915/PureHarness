import json
import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

tools = [
    {
        "type": "function",
        "name": "add",
        "description": "Add two integers and return the result.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {
                    "type": "integer",
                    "description": "The first integer.",
                },
                "b": {
                    "type": "integer",
                    "description": "The second integer.",
                },
            },
            "required": ["a", "b"],
        },
    }
]

response = client.responses.create(
    model="deepseek-v4-flash",
    input="Use the add tool to calculate 12 + 17.",
    tools=tools,
    tool_choice="required",
    reasoning={
        "effort": "none",
    },
)

print(response.to_json())