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


def pull_message(step_log: dict):
    if step_log.get("rationale"):
        yield ChatMessage(
            role="assistant", content=step_log["rationale"], thought=True
        )
    if step_log.get("tool_call"):
        used_code = step_log["tool_call"]["tool_name"] == "code interpreter"
        content = step_log["tool_call"]["tool_arguments"]
        if used_code:
            content = f"```py\n{content}\n```"
        yield ChatMessage(
            role="assistant",
            thought_metadata=ThoughtMetadata(
                tool_name=step_log["tool_call"]["tool_name"]
            ),
            content=content,
            thought=True,
        )
    if step_log.get("observation"):
        yield ChatMessage(
            role="assistant", content=step_log["observation"], thought=True
        )
    if step_log.get("error"):
        yield ChatMessage(
            role="assistant",
            content=str(step_log["error"]),
            thought=True,
            thought_metadata=ThoughtMetadata(error=True),
        )


def stream_from_transformers_agent(
    agent: Agent, prompt: str
) -> Generator[ChatMessage, None, None]:
    """Runs an agent with the given prompt and streams the messages from the agent as ChatMessages."""

    class Output:
        output: agent_types.AgentType | str = None

    for step_log in agent.run(prompt, stream=True):
        if isinstance(step_log, dict):
            for message in pull_message(step_log):
                yield message
    
    Output.output = step_log
    if isinstance(Output.output, agent_types.AgentText):
        yield ChatMessage(
            role="assistant", content=">>> " + Output.output.to_string(), thought=True
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
