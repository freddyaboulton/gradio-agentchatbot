import type { FileData } from "@gradio/client";

type OpenAIMessageRole = "system" | "user" | "assistant" | "tool";

export interface OpenAIMessage {
  role: OpenAIMessageRole;
  content: string;
  reasoning?: boolean;
  tool_name?: string | null;
  error?: boolean;
}

export interface FileMessage extends OpenAIMessage {
  file: FileData;
  alt_text?: string | null;
}