import type { FileData } from "@gradio/client";

type MessageRole = "system" | "user" | "assistant" | "tool";


export interface ThoughtMetadata {
  error: boolean;
  tool_name: string;
}

export interface Message {
  role: MessageRole;
  thought_metadata: ThoughtMetadata;
  thought: boolean;
}

export interface ChatMessage extends Message {
  content: string;
}

export interface ChatFileMessage extends Message {
  file: FileData;
  alt_text?: string | null;
}