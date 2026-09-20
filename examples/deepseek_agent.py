from dotenv import load_dotenv

from pureharness.agent import Agent
from pureharness.deepseek_model import DeepSeekModel
from pureharness.tools import ADD_TOOL, ToolRegistry


load_dotenv()

registry = ToolRegistry()
registry.register(ADD_TOOL)

model = DeepSeekModel()

agent = Agent(
    model=model,
    tools=registry,
)

result = agent.run(
    "Use the add tool to calculate 12 + 17."
)

print(result)