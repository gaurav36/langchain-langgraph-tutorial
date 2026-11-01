from langchain_core.prompts import ChatPromptTemplate
from global_llm_client import create_llm

llm = create_llm(temperature=0.4)

template = ChatPromptTemplate.from_messages([
    ("system", "You're a helpful programming assistant"),
    ("user", "{problem}")
])

# This pipe operator, pass the template to the llm. Its from the LCEL (LangChain Expression Language)
# Its goes from left to right, means output of template is passed as input to the llm
chain = template | llm # Create and run a chain

response = chain.stream({"problem": "Write a Python function to calculate the sigmoid function"})
for chunk in response:
    print(chunk.content, end="", flush=True)
print()
