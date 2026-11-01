from langchain_core.messages import SystemMessage, HumanMessage
from global_llm_client import create_llm

## this set temperature and max_output_token consumption, timeout, max_retries for the entire llm 
llm = create_llm(temperature=0.5, max_output_tokens=512, timeout=30, max_retries=3)


## Langchain have 4 message type
messages = [
    # SystemMessage: Sets behavior and context for the model. Its a system prompt. which tell model what to do. 
    SystemMessage(content="You're a helpful programming assistant"), 
    # HumanMessage: Represents user input like questions, commands, and data, e.g. what is recent changes in LangChain?
    HumanMessage(content="Write a Python code to make single neuron perceptron like model"), 
    # AIMessage: Contains model responses
    # ToolMessage: Results from tool/function calls (covered later)
]

response = llm.stream(messages)

## you can also enforce specific output constraints for a single call using bind().
## In this call, we limit the output length to 200 tokens.
# response = llm.bind(max_tokens=200).stream(messages) # limit output length just for this call
for chunk in response:
    print(chunk.content, end="", flush=True)
print()

"""
LangChain uses message objects instead of raw strings to represent conversational interactions.

Why Messages?
Structure: Clear separation of roles (system, user, assistant)
Metadata: Attach additional info (timestamps, sources, tool calls)
Type Safety: Easier to validate and debug
"""

# Messages can carry metadata. you can pass additional information like user ID, session ID, or any other context that might be useful for processing the message.
# This metadata can be useful for tracking, logging, or customizing the behavior of the model based on user-specific context.   
message_with_metadata = HumanMessage(
    content="Translate this to German: Hello, world!",
    additional_kwargs={"user_id": "12345-6789-1011-1213", "session": "abc-def-ghi-session1"}
)

print(f"invoking with metadata")
print("==============================\n")
response = llm.invoke([message_with_metadata])

# # Inspect response metadata
print("Content:", response.content)
print("\nResponse Metadata:")
print(f"  Model: {response.response_metadata.get('model_name')}")
print(f"  Tokens used: {response.response_metadata.get('token_usage')}")
print(f"  Finish reason: {response.response_metadata.get('finish_reason')}")
