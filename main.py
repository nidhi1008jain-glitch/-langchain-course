from dotenv import load_dotenv

load_dotenv()


from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from schemas import AgentResponse

from langchain_ollama import ChatOllama

tools = [TavilySearch()]
# llm = ChatOpenAI(model="gpt-4")
# llm = ChatOpenAI(model="gpt-5")
llm = ChatOllama(
    model="llama3.2:1b",   # or mistral, gemma, etc.
    temperature=0
)
# model = ChatOpenAI(model="gpt-4")
model = ChatOllama(model="llama3.2:1b")

agent = create_agent(
    model,
    tools=tools,
    response_format=AgentResponse
)

def main():
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details",
                }
            ]
        }
    )
      # Access structured response from the agent
    structured = result.get("structured_response", None)
    print(structured if structured is not None else result)


if __name__ == "__main__":
    main()
