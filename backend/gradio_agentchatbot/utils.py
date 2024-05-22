from transformers.agents import ReactAgent, agent_types
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


def convert_to_message_stream(message: dict) -> Generator[ChatMessage, None, None]:
    if message.get("rationale"):
        yield ChatMessage(
            role="assistant", content=message["rationale"]
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
        )
    if message.get("observation"):
        yield ChatMessage(
            role="assistant", content=message["observation"]
        )
    if message.get("error"):
        yield ChatMessage(
            role="assistant",
            content=str(message["error"]),
            thought_metadata=ThoughtMetadata(error=True),
        )


def pull_messages(new_messages: List[dict]):
    for message in new_messages:
        if not len(message):
            continue
        if message.get("rationale"):
            yield ChatMessage(
                role="assistant", content=message["rationale"]
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
            )
        if message.get("observation"):
            yield ChatMessage(
                role="assistant", content=message["observation"]
            )
        if message.get("error"):
            yield ChatMessage(
                role="assistant",
                content=str(message["error"]),
                thought_metadata=ThoughtMetadata(error=True),
            )


def stream_from_transformers_agent(
    agent: ReactAgent, prompt: str
) -> Generator[ChatMessage | ChatFileMessage, None, None]:
    """Runs an agent with the given prompt and streams the messages from the agent as ChatMessages."""

    for message in agent.run(prompt, stream=True):
        if isinstance(message, dict):
            for gradio_message in convert_to_message_stream(message):
                yield gradio_message
        elif isinstance(message, agent_types.AgentText):
            yield ChatMessage(
                role="assistant", content=message.to_string()
            )
        elif isinstance(message, agent_types.AgentImage):
            yield ChatFileMessage(
                role="assistant",
                file=FileData(path=message.to_string(), mime_type="image/png"),
                content="",
            )
        elif isinstance(message, agent_types.AgentAudio):
            yield ChatFileMessage(
                role="assistant",
                file=FileData(path=message.to_string(), mime_type="audio/wav"),
                content="",
            )
        elif isinstance(message, str):
            yield ChatMessage(role="assistant", content=message)
