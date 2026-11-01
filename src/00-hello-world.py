from global_llm_client import llm

response = llm.invoke("Tell me a joke about light bulbs!")
print(response.content)


"""
Key methods:
.invoke(input): single call
.batch(list_of_inputs): run many calls in parallel
.stream(input): (used later) stream tokens/chunks as they are generated

Why LangChain Wraps Models?
Portability: Switch between OpenAI, Groq, Anthropic with minimal code changes
Unified API: Same interface across all providers
Enhanced Features: Built-in retry logic, caching, tracing
"""
