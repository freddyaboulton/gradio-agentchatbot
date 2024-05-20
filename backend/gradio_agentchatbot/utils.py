from transformers.agents import Agent, agent_types
from pydantic import Field
from gradio.data_classes import GradioModel, FileData, GradioRootModel
from typing import Literal, List, Generator, Optional, Union
from threading import Thread
import time


class ThoughtMetadata(GradioModel):
    tool_name: Optional[str] = None
    error: bool = False


class Message(GradioModel):
    role: Literal["user", "assistant"]
    thought_metadata: ThoughtMetadata = Field(default_factory=ThoughtMetadata)


class ChatMessage(Message):
    content: str


class ChatFileMessage(Message):
    file: FileData
    alt_text: Optional[str] = None


class ChatbotData(GradioRootModel):
    root: List[Union[ChatMessage, ChatFileMessage]]


def pull_messages(new_messages: List[dict]):
    for message in new_messages:
        if not len(message):
            continue
        if message.get("rationale"):
            yield ChatMessage(
                role="assistant", content=message["rationale"], thought=True
            )
        if message.get("tool_call"):
            used_code = message["tool_call"]["tool_name"] == "code interpreter"
            content = message["tool_call"]["tool_arguments"]
            if used_code:
                content = f"```py\n{content}\n```"
            yield ChatMessage(
                role="assistant",
                thought_metadata=ThoughtMetadata(
                    tool_name=message["tool_call"]["tool_name"]
                ),
                content=content,
                thought=True,
            )
        if message.get("observation"):
            yield ChatMessage(
                role="assistant", content=message["observation"], thought=True
            )
        if message.get("error"):
            yield ChatMessage(
                role="assistant",
                content=str(message["error"]),
                thought=True,
                thought_metadata=ThoughtMetadata(error=True),
            )


def stream_from_transformers_agent(
    agent: Agent, prompt: str
) -> Generator[ChatMessage, None, None]:
    """Runs an agent with the given prompt and streams the messages from the agent as ChatMessages."""

    class Output:
        output: agent_types.AgentType | str = None

    def run_agent():
        output = agent.run(prompt)
        Output.output = output

    thread = Thread(target=run_agent)
    num_messages = 0

    # Start thread and pull logs while it runs
    thread.start()
    while thread.is_alive():
        if len(agent.logs) > num_messages:
            new_messages = agent.logs[num_messages:]
            for msg in pull_messages(new_messages):
                yield msg
                num_messages += 1
        time.sleep(0.1)

    thread.join(0.1)

    if len(agent.logs) > num_messages:
        new_messages = agent.logs[num_messages:]
        yield from pull_messages(new_messages)

    if isinstance(Output.output, agent_types.AgentText):
        yield ChatMessage(
            role="assistant", content=Output.output.to_string(), thought=True
        )
    elif isinstance(Output.output, agent_types.AgentImage):
        yield ChatFileMessage(
            role="assistant",
            file=FileData(path=Output.output.to_string(), mime_type="image/png"),
            content="",
            thought=True,
        )
    elif isinstance(Output.output, agent_types.AgentAudio):
        yield ChatFileMessage(
            role="assistant",
            file=FileData(path=Output.output.to_string(), mime_type="audio/wav"),
            content="",
            thought=True,
        )
    else:
        return ChatMessage(role="assistant", content=Output.output, thought=True)
