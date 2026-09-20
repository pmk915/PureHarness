from pureharness.messages import Message, ToolCall


ModelOutput = Message | list[ToolCall]