from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from typing import Annotated, Literal
import json
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from global_llm_client import create_llm

llm = create_llm(max_output_tokens=2056, temperature=0.8)

@tool
def calculator(
    # Annotations provide metadata for the model about the expected input
    # They help the model understand what kind of data it should provide for each parameter
    operation: Annotated[str, "The math operation: 'add', 'subtract', 'multiply', 'divide'"],
    a: Annotated[float, "First number"],
    b: Annotated[float, "Second number"]
) -> str:
    """Perform basic arithmetic operations. Use this when you need to calculate exact numerical results."""

    operations = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else "Error: Division by zero"
    }

    result = operations.get(operation.lower(), "Error: Invalid operation")
    return f"Result: {result}"

# what the tool looks like to the model
print(f"Tool name: {calculator.name}")
print(f"Tool description: {calculator.description}")
print(f"\nTool schema (what the model sees):")
print(json.dumps(calculator.get_input_schema().model_json_schema(), indent=2))

class WeatherInput(BaseModel):
    """Input for weather queries."""
    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit preference"
    )
    include_forecast: bool = Field(
        default=False,
        description="Include 5-day forecast"
    )

@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get current weather and optional forecast."""
    temp = 22 if units == "celsius" else 72
    result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days: Sunny"
    return result

# what the tool looks like to the model
print(f"Tool name: {get_weather.name}")
print(f"Tool description: {get_weather.description}")
print(f"\nTool schema (what the model sees):")
print(json.dumps(get_weather.get_input_schema().model_json_schema(), indent=2))

# Create our toolset
tools = [calculator, get_weather]

# Bind tools to the model
# This doesn't execute anything yet - just tells the model tools exist
llm_with_tools = llm.bind_tools(tools)

print(f"✓ Bound {len(tools)} tools to model")
print(f"Tools available: {[t.name for t in tools]}")

## Question that doesn't need tools
response = llm_with_tools.invoke([
    HumanMessage(content="What is the capital of Switzerland?")
])

print("Response type:", type(response).__name__)
print("Content:", response.content)
print("Tool calls:", response.tool_calls)  # Should be empty

# # Question requiring calculation
response = llm_with_tools.invoke([
    HumanMessage(content="What is 87 multiplied by 2?")
])

print("Response type:", type(response).__name__)
print("Content:", response.content)
print("\nTool calls made:")
print(json.dumps(response.tool_calls, indent=2))

## Question requiring weather tool
response = llm_with_tools.invoke([
    HumanMessage(content="What is the weather in Luzern?")
])

print("Response type:", type(response).__name__)
print("Content:", response.content)
print("\nTool calls made:")
print(json.dumps(response.tool_calls, indent=2))


"""
A tool in LangChain is:
- A Python function with metadata (name, description, schema)
- Something the LLM can "call" when it needs external capabilities
- Executed in your Python runtime, not by the model

Why tools matter:
- LLMs can't do math reliably → Give them a calculator tool
- LLMs can't access real-time data → Give them API tools
- LLMs can't execute code → Give them interpreter tools
- LLMs can't remember previous interactions → Give them memory tools
- LLMs can't search the web → Give them search tools

The @tool decorator automatically converts functions into LangChain tools.

Now comes the magic: we tell the model about our tools using bind_tools().

What happens behind the scenes:
- LangChain converts tool schemas to the format your model expects (OpenAI, Anthropic, etc.)
- The model can now "see" available tools in its context
- When needed, the model outputs a tool call instead of text

Important: Not all models support tool calling. Check your model's capabilities.

User asks a question → Model decides:
- Option A: Answer directly (no tools needed)
- Option B: Call one or more tools, then synthesize final answer
When the model chooses Option B, it returns an AIMessage with tool_calls instead of content.
"""

def run_tool_calling_flow(user_query: str):
    """
    Manual implementation of what agents do automatically.
    This helps understand the mechanics before we abstract it away.
    """

    print(f"\n{'='*60}")
    print(f"USER: {user_query}")
    print('='*60)

    # Step 1: Send query to model
    messages: list[BaseMessage] = [HumanMessage(content=user_query)]
    response = llm_with_tools.invoke(messages)

    # Step 2: Check if model wants to call tools
    if not response.tool_calls:
        print(f"\nMODEL (Direct answer): {response.content}")
        return response.content

    # Step 3: Model called tools - execute them
    print(f"\nMODEL: I need to call {len(response.tool_calls)} tool(s)...")
    messages.append(response)  # Add model's tool call to history

    # Create a mapping of tool names to actual functions
    tool_map = {t.name: t for t in tools}

    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        print(f"\n  → Calling: {tool_name}")
        print(f"    Arguments: {tool_args}")

        # Execute the tool
        selected_tool = tool_map[tool_name]
        tool_output = selected_tool.invoke(tool_args)

        print(f"    Result: {tool_output}")

        # Step 4: Add tool result to message history
        messages.append(ToolMessage(
            content=str(tool_output),
            tool_call_id=tool_id,
            name=tool_name
        ))

    # Step 5: Send everything back to model for final answer
    final_response = llm_with_tools.invoke(messages)
    print(f"\nMODEL (Final answer): {final_response.content}")

    return final_response.content

run_tool_calling_flow("What is half of 108 times 42?")
run_tool_calling_flow("Current weather in Luzern Switzerland? and next 5 days prediction? give result in table")
