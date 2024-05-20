---
tags: [gradio-custom-component, Chatbot, chatbot, agents, streaming, tools]
title: gradio_agentchatbot
short_description: Chat with agents 🤖 and see their thoughts 💭
colorFrom: blue
colorTo: yellow
sdk: gradio
pinned: false
app_file: app.py
---

# gradio_agentchatbot

## 🤖 Chat UI for displaying the thoughts of LLM Agents 💭
<img alt="Static Badge" src="https://img.shields.io/badge/version%20-%200.0.1%20-%20orange">  

The `gradio_agentchatbot` package introduces the `AgentChatbot` component which can display the thought process and tool usage of an LLM agent. Its message format is compatible with the OpenAI conversation message format. 

For example usage with transformers agents, please see the [Transformers Usage](#transformers-usage) section.

For general usage, see the [General Usage](#general-usage) section

For the API reference, see the [Initialization](#initialization) section.

## Installation

```bash
pip install gradio_agentchatbot
```

or add `gradio_agentchatbot` to your `requirements.txt`.

## Transformers Usage

For [transformers agents](https://huggingface.co/learn/cookbook/agents), you can use the `stream_from_transformers_agent` function and yield all subsequent messages.

```python

import gradio as gr
from transformers import load_tool, ReactCodeAgent, HfEngine, Tool
from gradio_agentchatbot import AgentChatbot, stream_from_transformers_agent, ChatMessage
from dotenv import load_dotenv
from langchain.agents import load_tools

# to load SerpAPI key
load_dotenv()

# Import tool from Hub
image_generation_tool = load_tool("m-ric/text-to-image")

search_tool = Tool.from_langchain(load_tools(["serpapi"])[0])

llm_engine = HfEngine("meta-llama/Meta-Llama-3-70B-Instruct")
# Initialize the agent with both tools
agent = ReactCodeAgent(tools=[image_generation_tool, search_tool], llm_engine=llm_engine)


def interact_with_agent(prompt, messages):
    messages.append(ChatMessage(role="user", content=prompt))
    yield messages
    for msg in stream_from_transformers_agent(agent, prompt):
        messages.append(msg)
        yield messages
    yield messages


with gr.Blocks() as demo:
    chatbot = AgentChatbot(label="Agent")
    text_input = gr.Textbox(lines=1, label="Chat Message")
    text_input.submit(interact_with_agent, [text_input, chatbot], [chatbot])


if __name__ == "__main__":
    demo.launch()
```

![AgentChatbot with transformers](https://gradio-builds.s3.amazonaws.com/demo-files/tf_agent_ui.gif)

## General Usage

The `AgentChatbot` is similar to the core `Gradio` `Chatbot` but the key difference is in the expected data format of the `value` property.

Instead of a list of tuples, each of which can be either a string or tuple, the value is a list of message instances. Each message can be either a `ChatMessage` or a `ChatFileMessage`. These are pydantic classes that are compatible with the OpenAI [message format](https://platform.openai.com/docs/api-reference/chat/create#chat-create-messages). This is how they are defined:

```python
class ThoughtMetadata(GradioModel):
    tool_name: Optional[str] = None
    error: bool = False


class ChatMessage(GradioModel):
    role: Literal["user", "assistant"]
    content: str
    thought_metadata: ThoughtMetadata = Field(default_factory=ThoughtMetadata)


class ChatFileMessage(GradioModel):
    role: Literal["user", "assistant"]
    file: FileData
    thought_metadata: ThoughtMetadata = Field(default_factory=ThoughtMetadata)
    alt_text: Optional[str] = None
```

In order to properly display data in `AgentChatbot`, simply return a list of `ChatMessage` or `ChatFileMessage` instances from your python function. For example:

```python
def chat_echo(prompt: str, messages: List[ChatMessage | ChatFileMessage]) -> List[ChatMessage | ChatFileMessage]:
    messages.append(ChatMessage(role="user", content=prompt))
    messages.append(ChatMessage(role="assistant", content=prompt))
    return messages
```

### Why a different data format than Gradio core?

The OpenAI data format is the standard format for representing LLM conversations and most API providers have adopted it.
By using a compliant data format, it should be easier to use `AgentChatbot` with multiple API providers and libraries.


### What is `thought_metadata` field for?

You can use this to add additional information data about the current thought, like the names of the tool used.
If the `thought_metadata.tool_name` field is not `None`, the message `content` will be displayed in a collapsible tool modal. See below:

![Tool Modal](https://gradio-builds.s3.amazonaws.com/demo-files/tool_modal.gif)


### Why are pydantic data classes required?

It should improve developer experience since your editor will auto-complete the required fields and use smart autocomplete for the `role` class. You will also get an error message if your data does not conform to the data format.

I will probably relax this in the future so that a plain python dict can be passed instead of one of the chat classes.



## `API Reference`

### Initialization

<table>
<thead>
<tr>
<th align="left">name</th>
<th align="left" style="width: 25%;">type</th>
<th align="left">default</th>
<th align="left">description</th>
</tr>
</thead>
<tbody>
<tr>
<td align="left"><code>value</code></td>
<td align="left" style="width: 25%;">

```python
list[ChatMessage | ChatFileMessage]
    | Callable
    | None
```

</td>
<td align="left"><code>None</code></td>
<td align="left">Default value to show in chatbot. If callable, the function will be called whenever the app loads to set the initial value of the component.</td>
</tr>

<tr>
<td align="left"><code>label</code></td>
<td align="left" style="width: 25%;">

```python
str | None
```

</td>
<td align="left"><code>None</code></td>
<td align="left">The label for this component. Appears above the component and is also used as the header if there are a table of examples for this component. If None and used in a `gr.Interface`, the label will be the name of the parameter this component is assigned to.</td>
</tr>

<tr>
<td align="left"><code>every</code></td>
<td align="left" style="width: 25%;">

```python
float | None
```

</td>
<td align="left"><code>None</code></td>
<td align="left">If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.</td>
</tr>

<tr>
<td align="left"><code>show_label</code></td>
<td align="left" style="width: 25%;">

```python
bool | None
```

</td>
<td align="left"><code>None</code></td>
<td align="left">if True, will display label.</td>
</tr>

<tr>
<td align="left"><code>container</code></td>
<td align="left" style="width: 25%;">

```python
bool
```

</td>
<td align="left"><code>True</code></td>
<td align="left">If True, will place the component in a container - providing some extra padding around the border.</td>
</tr>

<tr>
<td align="left"><code>scale</code></td>
<td align="left" style="width: 25%;">

```python
int | None
```

</td>
<td align="left"><code>None</code></td>
<td align="left">relative size compared to adjacent Components. For example if Components A and B are in a Row, and A has scale=2, and B has scale=1, A will be twice as wide as B. Should be an integer. scale applies in Rows, and to top-level Components in Blocks where fill_height=True.</td>
</tr>

<tr>
<td align="left"><code>min_width</code></td>
<td align="left" style="width: 25%;">

```python
int
```

</td>
<td align="left"><code>160</code></td>
<td align="left">minimum pixel width, will wrap if not sufficient screen space to satisfy this value. If a certain scale value results in this Component being narrower than min_width, the min_width parameter will be respected first.</td>
</tr>

<tr>
<td align="left"><code>visible</code></td>
<td align="left" style="width: 25%;">

```python
bool
```

</td>
<td align="left"><code>True</code></td>
<td align="left">If False, component will be hidden.</td>
</tr>

<tr>
<td align="left"><code>elem_id</code></td>
<td align="left" style="width: 25%;">

```python
str | None
```

</td>
<td align="left"><code>None</code></td>
<td align="left">An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.</td>
</tr>

<tr>
<td align="left"><code>elem_classes</code></td>
<td align="left" style="width: 25%;">

```python
list[str] | str | None
```

</td>
<td align="left"><code>None</code></td>
<td align="left">An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.</td>
</tr>

<tr>
<td align="left"><code>render</code></td>
<td align="left" style="width: 25%;">

```python
bool
```

</td>
<td align="left"><code>True</code></td>
<td align="left">If False, component will not render be rendered in the Blocks context. Should be used if the intention is to assign event listeners now but render the component later.</td>
</tr>

<tr>
<td align="left"><code>key</code></td>
<td align="left" style="width: 25%;">

```python
int | str | None
```

</td>
<td align="left"><code>None</code></td>
<td align="left">if assigned, will be used to assume identity across a re-render. Components that have the same key across a re-render will have their value preserved.</td>
</tr>

<tr>
<td align="left"><code>height</code></td>
<td align="left" style="width: 25%;">

```python
int | str | None
```

</td>
<td align="left"><code>None</code></td>
<td align="left">The height of the component, specified in pixels if a number is passed, or in CSS units if a string is passed.</td>
</tr>

<tr>
<td align="left"><code>latex_delimiters</code></td>
<td align="left" style="width: 25%;">

```python
list[dict[str, str | bool]] | None
```

</td>
<td align="left"><code>None</code></td>
<td align="left">A list of dicts of the form {"left": open delimiter (str), "right": close delimiter (str), "display": whether to display in newline (bool)} that will be used to render LaTeX expressions. If not provided, `latex_delimiters` is set to `[{ "left": "$$", "right": "$$", "display": True }]`, so only expressions enclosed in $$ delimiters will be rendered as LaTeX, and in a new line. Pass in an empty list to disable LaTeX rendering. For more information, see the [KaTeX documentation](https://katex.org/docs/autorender.html).</td>
</tr>

<tr>
<td align="left"><code>rtl</code></td>
<td align="left" style="width: 25%;">

```python
bool
```

</td>
<td align="left"><code>False</code></td>
<td align="left">If True, sets the direction of the rendered text to right-to-left. Default is False, which renders text left-to-right.</td>
</tr>

<tr>
<td align="left"><code>show_share_button</code></td>
<td align="left" style="width: 25%;">

```python
bool | None
```

</td>
<td align="left"><code>None</code></td>
<td align="left">If True, will show a share icon in the corner of the component that allows user to share outputs to Hugging Face Spaces Discussions. If False, icon does not appear. If set to None (default behavior), then the icon appears if this Gradio app is launched on Spaces, but not otherwise.</td>
</tr>

<tr>
<td align="left"><code>show_copy_button</code></td>
<td align="left" style="width: 25%;">

```python
bool
```

</td>
<td align="left"><code>False</code></td>
<td align="left">If True, will show a copy button for each chatbot message.</td>
</tr>

<tr>
<td align="left"><code>avatar_images</code></td>
<td align="left" style="width: 25%;">

```python
tuple[str | Path | None, str | Path | None] | None
```

</td>
<td align="left"><code>None</code></td>
<td align="left">Tuple of two avatar image paths or URLs for user and bot (in that order). Pass None for either the user or bot image to skip. Must be within the working directory of the Gradio app or an external URL.</td>
</tr>

<tr>
<td align="left"><code>sanitize_html</code></td>
<td align="left" style="width: 25%;">

```python
bool
```

</td>
<td align="left"><code>True</code></td>
<td align="left">If False, will disable HTML sanitization for chatbot messages. This is not recommended, as it can lead to security vulnerabilities.</td>
</tr>

<tr>
<td align="left"><code>render_markdown</code></td>
<td align="left" style="width: 25%;">

```python
bool
```

</td>
<td align="left"><code>True</code></td>
<td align="left">If False, will disable Markdown rendering for chatbot messages.</td>
</tr>

<tr>
<td align="left"><code>bubble_full_width</code></td>
<td align="left" style="width: 25%;">

```python
bool
```

</td>
<td align="left"><code>True</code></td>
<td align="left">If False, the chat bubble will fit to the content of the message. If True (default), the chat bubble will be the full width of the component.</td>
</tr>

<tr>
<td align="left"><code>line_breaks</code></td>
<td align="left" style="width: 25%;">

```python
bool
```

</td>
<td align="left"><code>True</code></td>
<td align="left">If True (default), will enable Github-flavored Markdown line breaks in chatbot messages. If False, single new lines will be ignored. Only applies if `render_markdown` is True.</td>
</tr>

<tr>
<td align="left"><code>likeable</code></td>
<td align="left" style="width: 25%;">

```python
bool
```

</td>
<td align="left"><code>False</code></td>
<td align="left">Whether the chat messages display a like or dislike button. Set automatically by the .like method but has to be present in the signature for it to show up in the config.</td>
</tr>

<tr>
<td align="left"><code>layout</code></td>
<td align="left" style="width: 25%;">

```python
Literal["panel", "bubble"] | None
```

</td>
<td align="left"><code>None</code></td>
<td align="left">If "panel", will display the chatbot in a llm style layout. If "bubble", will display the chatbot with message bubbles, with the user and bot messages on alterating sides. Will default to "bubble".</td>
</tr>

<tr>
<td align="left"><code>placeholder</code></td>
<td align="left" style="width: 25%;">

```python
str | None
```

</td>
<td align="left"><code>None</code></td>
<td align="left">a placeholder message to display in the chatbot when it is empty. Centered vertically and horizontally in the TestChatbot. Supports Markdown and HTML. If None, no placeholder is displayed.</td>
</tr>
</tbody></table>


### Events

| name | description |
|:-----|:------------|
| `change` | Triggered when the value of the TestChatbot changes either because of user input (e.g. a user types in a textbox) OR because of a function update (e.g. an image receives a value from the output of an event trigger). See `.input()` for a listener that is only triggered by user input. | 