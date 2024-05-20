import gradio as gr
from transformers import load_tool, ReactCodeAgent, HfEngine, Tool
from gradio_agentchatbot import AgentChatbot, stream_from_agent, OpenAIMessage
from dotenv import load_dotenv
from langchain.agents import load_tools

load_dotenv()

# Import tool from Hub
image_generation_tool = load_tool("m-ric/text-to-image")

search_tool = Tool.from_langchain(load_tools(["serpapi"])[0])

llm_engine = HfEngine("meta-llama/Meta-Llama-3-70B-Instruct")
# Initialize the agent with both tools
agent = ReactCodeAgent(tools=[image_generation_tool, search_tool], llm_engine=llm_engine)


def interact_with_agent(prompt, messages):
    messages.append(OpenAIMessage(role="user", content=prompt))
    yield messages, None
    for msg in stream_from_agent(agent, prompt):
        messages.append(msg)
        yield messages, None
    yield messages, agent.logs

with gr.Blocks() as demo:
    chatbot = AgentChatbot()
    text_input = gr.Textbox(lines=1, label="Input")
    json_output = gr.Json(label="Output")
    text_input.submit(interact_with_agent, [text_input, chatbot], [chatbot, json_output])


if __name__ == "__main__":
    demo.launch()
