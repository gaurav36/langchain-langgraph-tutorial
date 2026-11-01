from global_llm_client import llm

response = llm.stream("What is LangChain?")
for chunk in response:
    print(chunk.content, end="", flush=True)
print()
