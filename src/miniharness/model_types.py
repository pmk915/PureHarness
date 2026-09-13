from miniharness.messages import Message, ToolCall


ModelOutput = Message | list[ToolCall]