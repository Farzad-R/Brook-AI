"""In this module, we define helper functions to pretty print the messages in the graph while we debug it and to give our tool node error handling (by adding the error to the chat history).

Functions include:
1. `handle_tool_error`: Formats error messages for display and adds them to the chat history.
2. `create_tool_node_with_fallback`: Creates a `ToolNode` with error handling fallback logic.
3. `_print_event`: Prints the current state and messages of an event, with optional truncation for long messages.
"""

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode


def handle_tool_error(state) -> dict:
    """
    Handles errors by formatting them into a message and adding them to the chat history.

    This function retrieves the error from the given state and formats it into a `ToolMessage`, which is then
    added to the chat history. It uses the latest tool calls from the state to attach the error message.

    Args:
        state (dict): The current state of the tool, which includes error information and tool calls.

    Returns:
        dict: A dictionary containing a list of `ToolMessage` objects with error information.
    """
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    """
    Creates a `ToolNode` with fallback error handling.

    This function creates a `ToolNode` object and configures it to use a fallback function for error handling. 
    The fallback function handles errors by calling `handle_tool_error`.

    Args:
        tools (list): A list of tools to be included in the `ToolNode`.

    Returns:
        dict: A `ToolNode` configured with fallback error handling.
    """
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def _print_event(event: dict, _printed: set, max_length=1500):
    """
    Prints the current state and messages of an event, with optional truncation for long messages.

    This function prints information about the current dialog state and the latest message in the event. If the message 
    is too long, it is truncated to a specified maximum length.

    Args:
        event (dict): The event containing dialog state and messages.
        _printed (set): A set of message IDs that have already been printed, to avoid duplicate output.
        max_length (int, optional): The maximum length of the message to print before truncating. Defaults to 1500.
    """
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)
