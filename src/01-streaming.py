from global_llm_client import create_llm

llm = create_llm(max_output_tokens=512, temperature=0.8)

response = llm.stream("What is LangChain?")
for chunk in response:
    print(chunk.content, end="", flush=True)
print()
