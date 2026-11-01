from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# Template are resuable set of objects. They allow you to define a structure for your prompts and reuse them with
# different inputs.
prompt_template = ChatPromptTemplate.from_template('tell me a joke about Trump')
print(prompt_template.format_messages())

## It will give following output:
""""
[HumanMessage(content='tell me a joke about Trump', additional_kwargs={}, response_metadata={})]
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an English to German translator."),
    ("user", "Translate this to German: {text}")
])
print(prompt_template.format_messages(text="Hello, how are you?"))

## Above will print the formatted messages with the provided input text.
"""
[SystemMessage(content='You are an English to German translator.', additional_kwargs={}, response_metadata={}),
HumanMessage(content='Translate this to German: Hello, how are you?', additional_kwargs={}, response_metadata={})]
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You're a helpful programming assistant"),
    ("user", "{problem}")
])

python_problem_statement = "Write a Python function to calculate factorial"

print(prompt_template.format_messages(problem=python_problem_statement))

messages = [SystemMessage(content="You're a helpful programming assistant")]
messages.append(HumanMessage(content=python_problem_statement))
print(messages)
print(messages == prompt_template.format_messages(problem=python_problem_statement))
