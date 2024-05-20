from transformers.agents import Agent, agent_types
from gradio.data_classes import GradioModel, FileData, GradioRootModel
from typing import Literal, List, Generator, Optional, Union
from threading import Thread
import time


class OpenAIMessage(GradioModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    reasoning: bool = False
    tool_name: Optional[str] = None
    error: bool = False
    thoughts: list["OpenAIMessage"]


class FileMessage(OpenAIMessage):
    file: FileData
    alt_text: Optional[str] = None


class ChatbotData(GradioRootModel):
    root: List[Union[OpenAIMessage, FileMessage]]


def pull_messages(new_messages: List[dict]):
    for message in new_messages:
        if not len(message):
            continue
        if message.get("rationale"):
            yield OpenAIMessage(
                role="assistant", content=message["rationale"], reasoning=True
            )
        if message.get("tool_call"):
            yield OpenAIMessage(
                role="assistant",
                tool_name=message["tool_call"]["tool_name"],
                content=message["tool_call"]["tool_arguments"],
                reasoning=True,
            )
        if message.get("observation"):
            yield OpenAIMessage(
                role="assistant", content=message["observation"], reasoning=True
            )
        if message.get("error"):
            yield OpenAIMessage(
                role="assistant", content=str(message["error"]), error=True, reasoning=True
            )


def stream_from_agent(
    agent: Agent, prompt: str
) -> Generator[OpenAIMessage, None, None]:
    
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
        yield OpenAIMessage(role="assistant", content=Output.output.to_string(), reasoning=True)
    elif isinstance(Output.output, agent_types.AgentImage):
        yield FileMessage(
            role="assistant",
            file=FileData(path=Output.output.to_string(), mime_type="image/png"),
            content="",
            reasoning=True,
        )
    elif isinstance(Output.output, agent_types.AgentAudio):
        yield FileMessage(
            role="assistant",
            file=FileData(path=Output.output.to_string(), mime_type="audio/wav"),
            content="",
            reasoning=True,
        )
    else:
        return OpenAIMessage(role="assistant", content=Output.output, reasoning=True)
