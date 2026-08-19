"""Conversation mode: characters, chat history, and the prompt built from them.

Nothing here touches ``prompt_engine``. Prompt mode's output is a vendored
engine's, verified against upstream byte-for-byte; a conversation is this
application's own, and keeping the two apart is what lets the chat system
prompt change without a parity report having to be re-run.
"""
