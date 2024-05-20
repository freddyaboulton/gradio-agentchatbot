import gradio as gr
from transformers import load_tool, ReactCodeAgent, HfEngine, Tool
from gradio_agentchatbot import (
    AgentChatbot,
    stream_from_transformers_agent,
    ChatMessage,
)
from dotenv import load_dotenv
from langchain.agents import load_tools
from pathlib import Path

current_dir = Path(__file__).parent

load_dotenv()

# Import tool from Hub
image_generation_tool = load_tool("m-ric/text-to-image")

search_tool = Tool.from_langchain(load_tools(["serpapi"])[0])

llm_engine = HfEngine("meta-llama/Meta-Llama-3-70B-Instruct")
# Initialize the agent with both tools
agent = ReactCodeAgent(
    tools=[image_generation_tool, search_tool], llm_engine=llm_engine
)


def interact_with_agent(prompt, messages):
    messages.append(ChatMessage(role="user", content=prompt))
    yield messages
    for msg in stream_from_transformers_agent(agent, prompt):
        messages.append(msg)
        yield messages
    yield messages


with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.Tab("Demo"):
            gr.Markdown("# Chat with an LLM Agent 🤖 and see its thoughts 💭")
            chatbot = AgentChatbot(
                label="Agent",
                avatar_images=[
                    None,
                    "https://em-content.zobj.net/source/twitter/53/robot-face_1f916.png",
                ],
            )
            text_input = gr.Textbox(lines=1, label="Chat Message")
            text_input.submit(interact_with_agent, [text_input, chatbot], [chatbot])
        with gr.Tab("Docs"):
            gr.Markdown(Path(current_dir / "docs.md").read_text())


if __name__ == "__main__":
    demo.launch()
