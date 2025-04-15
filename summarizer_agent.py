import os
import re
from dotenv import load_dotenv
import operator
import ast # For safe evaluation of string literals
import json # For potentially parsing JSON strings
from typing import TypedDict, Annotated, Sequence, List, Optional, Any

# --- LangChain Imports ---
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document # Import Document type

# --- LangGraph Imports ---
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

# --- Configuration ---
load_dotenv()
MAX_CHARS_FOR_STUFF = 15000 # Rough estimate, adjust based on model context window & typical overhead

# --- Agent State ---
# Expanded state to manage the multi-step process
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_input: str # Store the original user request (URL + format)
    url: Optional[str] # Make optional initially
    # Store documents directly, or combined text. Text is simpler for length check.
    fetched_text: Optional[str]
    fetch_error: Optional[str] # Flag if fetching failed
    determined_chain_type: Optional[str]
    raw_summary: Optional[str]
    final_summary: Optional[str] # The potentially formatted summary
    # Add formatter_llm instance to state if needed across nodes, or pass differently
    # For simplicity, let's recreate it in the node if needed, or pass via config
    # config: Optional[dict] # Alternative way to pass non-state info like LLMs


# --- Define Tools ---

@tool
def fetch_content_tool(url: str) -> dict:
    """
    Fetches text content from a given URL.
    Returns a dictionary containing 'text' (the concatenated page content)
    or 'error' if fetching fails.
    """
    print(f"--- Calling Fetch Content Tool for: {url} ---")
    if not url or not isinstance(url, str):
         return {"error": "Invalid URL provided to fetch_content_tool."}
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        if not docs:
            return {"error": f"Could not load any content from the URL: {url}"}

        # Filter out docs with empty page_content before joining
        page_contents = [doc.page_content for doc in docs if doc.page_content and isinstance(doc.page_content, str)]
        if not page_contents:
             return {"error": f"Loaded document content is empty for URL: {url}"}

        full_text = "\n\n".join(page_contents) # Use double newline as separator

        print(f"--- Fetch Content Tool Success (length: {len(full_text)}) ---")
        return {"text": full_text}
    except Exception as e:
        print(f"--- Fetch Content Tool Error: {e} ---")
        # Return a serializable error message
        return {"error": f"An error occurred while fetching content from {url}: {str(e)}"}

@tool
def summarize_content_tool(text_to_summarize: str, chain_type: str, model_name: str = "gemini-pro") -> dict:
    """
    Summarizes the provided text using a specified LangChain chain type ('stuff', 'map_reduce', 'refine')
    and a specific Google Gemini model.
    Use this *after* content has been fetched and the appropriate chain_type has been determined.
    Input requires 'text_to_summarize', 'chain_type', and optionally 'model_name'.
    Returns a dictionary containing 'summary' or 'error'.
    """
    print(f"--- Calling Summarize Content Tool (Type: {chain_type}, Model: {model_name}) ---")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return {"error": "GOOGLE_API_KEY not found."}
    if not text_to_summarize or not isinstance(text_to_summarize, str) or not text_to_summarize.strip():
         return {"error": "No valid text provided to summarize."}
    if not chain_type or chain_type not in ["stuff", "map_reduce", "refine"]:
         return {"error": f"Invalid chain_type '{chain_type}' provided."}

    try:
        # Initialize LLM for summarization
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.1, # Slightly higher temp for potentially more natural summaries
            convert_system_message_to_human=True
        ) # <-- Fixed: Added missing closing parenthesis

        # Prepare documents for the chain
        docs = [Document(page_content=text_to_summarize)]

        # Split if using map_reduce or refine
        if chain_type in ["map_reduce", "refine"]:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200,
                length_function=len
            )
            split_docs = text_splitter.split_documents(docs)
            print(f"--- Splitting text for '{chain_type}'. Number of chunks: {len(split_docs)} ---")
            # Handle case where splitting results in empty list (shouldn't happen with valid input)
            if not split_docs:
                 return {"error": "Text splitting resulted in no processable chunks."}
            docs_to_process = split_docs
        else: # 'stuff'
             docs_to_process = docs

        # Load and run the summarization chain
        chain = load_summarize_chain(llm, chain_type=chain_type, verbose=False)
        result = chain.invoke({"input_documents": docs_to_process}) # Use invoke directly
        summary = result.get("output_text")
        if not summary or not isinstance(summary, str):
             # Handle potential chain errors or empty summaries
             print(f"--- Summarization chain did not return valid output text. Result: {result} ---")
             return {"error": "Summarization failed to produce text output."}


        print(f"--- Summarize Content Tool Success ---")
        return {"summary": summary.strip()}

    except Exception as e:
        print(f"--- Summarize Content Tool Error: {e} ---")
        # Ensure error message is serializable
        return {"error": f"An error occurred during summarization ({chain_type}): {str(e)}"}


# List of tools available to the agent LLM
tools = [fetch_content_tool, summarize_content_tool]

# --- Agent Definition ---
# LLMs will be created in build_graph

# --- Graph Nodes ---

# 1. Decide initial action (fetch or respond if no URL)
# REVISED VERSION - Returning Updates Dictionary
def route_initial_request(state: AgentState) -> dict:
    """
    Determines the first step based on user input.
    Returns a dictionary containing state updates ('url', 'user_input',
    'fetch_error', 'final_summary').
    """
    print("--- Routing Initial Request ---")
    updates: dict[str, Any] = {} # Initialize dictionary for updates

    messages = state.get('messages', [])
    if not messages:
         print("--- Error: No messages found in state. ---")
         updates['final_summary'] = "Internal error: Agent state is missing messages."
         updates['fetch_error'] = updates['final_summary'] # Signal error
         return updates # Return the updates dict

    last_message_content = messages[-1].content
    if not isinstance(last_message_content, str):
        print(f"--- Error: Last message content is not a string ({type(last_message_content)}). ---")
        updates['final_summary'] = "Internal error: Invalid user input type."
        updates['fetch_error'] = updates['final_summary'] # Signal error
        return updates # Return the updates dict

    user_input = last_message_content
    url_match = re.search(r'https?://\S+', user_input)

    if url_match:
        url = url_match.group(0)
        print(f"--- URL Found: {url}. Proceeding to fetch. ---")
        # Add changes to the updates dictionary
        updates['url'] = url
        updates['user_input'] = user_input
        # Optional: Add back the intent message if desired
        # fetch_intent_message = AIMessage(content=f"Okay, I will fetch content from {url}...", tool_calls=[])
        # updates['messages'] = [fetch_intent_message] # This will be added via operator.add
    else:
        print("--- No URL found in input. Responding directly. ---")
        # Add changes to the updates dictionary
        updates['final_summary'] = "Please provide a valid URL for me to summarize."
        updates['fetch_error'] = updates['final_summary'] # Signal error/end

    # Return the dictionary containing all updates for LangGraph to merge
    print(f"--- initial_router returning updates: {updates} ---") # Debug: See what's returned
    return updates

# 1. Decide initial action (fetch or respond if no URL)
# MODIFIED VERSION
def route_initial_request(state: AgentState) -> None: # Return None explicitly
    """
    Determines the first step based on user input.
    Modifies state dictionary IN PLACE with URL, user_input, or final_summary/fetch_error.
    The actual routing happens in add_conditional_edges based on state['url'].
    """
    print("--- Routing Initial Request ---")
    messages = state.get('messages', []) # Use .get for safety
    if not messages:
         print("--- Error: No messages found in state. ---")
         # Modify state directly
         state['final_summary'] = "Internal error: Agent state is missing messages."
         state['fetch_error'] = state['final_summary']
         return # Exit function

    # Check type of last message content
    last_message_content = messages[-1].content
    if not isinstance(last_message_content, str):
        print(f"--- Error: Last message content is not a string ({type(last_message_content)}). ---")
        # Modify state directly
        state['final_summary'] = "Internal error: Invalid user input type."
        state['fetch_error'] = state['final_summary']
        return # Exit function

    user_input = last_message_content
    url_match = re.search(r'https?://\S+', user_input)

    if url_match:
        url = url_match.group(0)
        # Modify state dictionary directly
        state['url'] = url
        state['user_input'] = user_input
        print(f"--- URL Found: {url}. Proceeding to fetch. ---")
        # Removed adding the AIMessage here - node focuses on setting state for routing
    else:
        print("--- No URL found in input. Responding directly. ---")
         # Modify state dictionary directly
        state['final_summary'] = "Please provide a valid URL for me to summarize."
        state['fetch_error'] = state['final_summary'] # Signal error/end

    # The function modifies state in place, no return value needed for graph flow here
    return None

# 2. Node to call the fetch_content_tool
def call_fetcher(state: AgentState) -> dict:
    """
    Prepares the call and invokes the fetch_content_tool using ToolNode.
    Returns state updates (fetched_text or fetch_error, messages).
    """
    print("--- Calling Fetcher Node ---")
    url_to_fetch = state.get('url')
    if not url_to_fetch:
         print("--- Fetcher Node Error: URL missing in state. ---")
         # This shouldn't happen if routing is correct, but handle defensively
         return {"fetch_error": "URL missing in state", "final_summary": "Internal Error: URL not found."}

    # Construct the AIMessage that simulates the LLM deciding to call the tool
    fetch_tool_call = AIMessage(
        content="", # No textual content needed from AI here
        tool_calls=[{
            "name": fetch_content_tool.name,
            "args": {"url": url_to_fetch},
            "id": "fetch_call_001" # Make ID unique if needed, simple ID is fine
        }]
    )

    # The ToolNode expects the state's messages list ending with the AIMessage containing tool_calls
    # We pass *only* this required structure to invoke, not the whole state message history
    fetch_tool_node = ToolNode([fetch_content_tool])
    tool_result_message = fetch_tool_node.invoke({"messages": [fetch_tool_call]})

    # Process the ToolMessage result
    updates_to_state = {"messages": [tool_result_message]} # Always add the tool message to history
    if isinstance(tool_result_message, ToolMessage) and tool_result_message.content:
        try:
            # Use ast.literal_eval for safety, assuming dict-like string output
            # If tool returns actual JSON string, use json.loads(tool_result_message.content)
            result_data = ast.literal_eval(tool_result_message.content)
            # result_data = json.loads(tool_result_message.content) # Alternative if JSON

            if isinstance(result_data, dict):
                if "error" in result_data:
                    error_msg = result_data['error']
                    print(f"--- Fetcher Node Error reported by tool: {error_msg} ---")
                    updates_to_state["fetch_error"] = error_msg
                    updates_to_state["final_summary"] = f"Failed to fetch content: {error_msg}"
                elif "text" in result_data:
                    print("--- Fetcher Node Success ---")
                    updates_to_state["fetched_text"] = result_data['text']
                else:
                    print("--- Fetcher Node Error: Tool output missing 'text' or 'error'. ---")
                    updates_to_state["fetch_error"] = "Tool output missing 'text' or 'error'."
                    updates_to_state["final_summary"] = "Internal Error: Fetcher tool returned unexpected data."
            else:
                 print("--- Fetcher Node Error: Parsed tool output is not a dictionary. ---")
                 updates_to_state["fetch_error"] = "Parsed tool output is not a dictionary."
                 updates_to_state["final_summary"] = "Internal Error: Fetcher tool returned unexpected data format."

        except (SyntaxError, ValueError, TypeError) as e:
            # Handle errors during parsing (e.g., if content is not a valid literal/JSON)
            print(f"--- Error processing fetcher result string: {e} ---")
            error_msg = f"Error processing fetcher result: {e}"
            updates_to_state["fetch_error"] = error_msg
            updates_to_state["final_summary"] = error_msg
    else:
         # Handle cases where tool_result_message is not as expected
         print("--- Fetcher Node returned unexpected result type or empty content. ---")
         error_msg = "Fetcher tool did not return expected output."
         updates_to_state["fetch_error"] = error_msg
         updates_to_state["final_summary"] = error_msg

    return updates_to_state # Return dictionary of updates


# 3. Decide summarization chain type based on content length
def decide_chain_type_node(state: AgentState) -> dict:
    """
    Decides chain type based on fetched text length.
    Returns state updates (determined_chain_type or fetch_error/final_summary).
    """
    print("--- Deciding Chain Type Node ---")
    # This node runs *after* call_fetcher, check if fetch_error was set
    if state.get('fetch_error'):
        print("--- Skipping chain type decision due to fetch error. ---")
        # No updates needed, error already set
        return {}

    fetched_text = state.get('fetched_text')
    if not fetched_text or not isinstance(fetched_text, str):
        print("--- No valid fetched text found. Cannot decide chain type. ---")
        # Update state with error
        error_msg = "Error: Content was fetched but is missing or invalid."
        return {"fetch_error": error_msg, "final_summary": error_msg}

    text_length = len(fetched_text)
    if text_length < MAX_CHARS_FOR_STUFF:
        chain_type = "stuff"
    else:
        chain_type = "map_reduce" # Default to map_reduce for longer content

    print(f"--- Determined Chain Type: {chain_type} (Length: {text_length}) ---")
    # Return updates for the state
    return {"determined_chain_type": chain_type}

# 4. Node to call the Summarizer Agent/Tool
# This node will invoke the main agent LLM, asking it to *use* the summarize tool
def call_summarizer_agent(state: AgentState, agent_executor) -> dict:
    """Invokes the agent LLM, prompting it to use the summarize tool."""
    print("--- Calling Summarizer Agent Node ---")
    if state.get('fetch_error'):
        print("--- Skipping summarization due to fetch error. ---")
        # Return final error message as AI response to add to history and end
        return {"messages": [AIMessage(content=state['final_summary'], tool_calls=[])]}

    fetched_text_len = len(state.get('fetched_text', ''))
    chain_type = state.get('determined_chain_type', 'N/A')
    original_request = state.get('user_input', '')

    # Construct the prompt message for the agent
    prompt_message = HumanMessage(
        content=f"Okay, I have fetched the content (length: {fetched_text_len}). "
                f"The determined summarization type is '{chain_type}'. "
                f"Please now summarize the content using the 'summarize_content_tool'. "
                f"Remember the original request for context: '{original_request}'"
    )

    # Invoke the agent executor with the current messages + new prompt
    # The agent executor handles adding the prompt to the history internally
    response = agent_executor.invoke({"messages": state['messages'] + [prompt_message]})

    print(f"--- Summarizer Agent LLM Response (raw): {response} ---") # Debug raw response

    # The agent_executor's response format might vary, typically includes 'messages' or 'output'
    # We expect an AIMessage with tool calls here. Extract it carefully.
    agent_response_message = None
    if isinstance(response, dict) and "messages" in response and response["messages"]:
         # If the executor returns the full state including messages
         agent_response_message = response["messages"][-1] # Get the last message added by the agent
    elif isinstance(response, dict) and "output" in response:
         # Handle cases where output might be structured differently
         # This might need adjustment based on the specific agent executor used
         # Assuming output contains the AIMessage or its content string
         output_content = response["output"]
         if isinstance(output_content, AIMessage):
             agent_response_message = output_content
         elif isinstance(output_content, str): # If just the text content is returned
              # We need the AIMessage object, potentially with tool calls. This might be insufficient.
              # Let's assume for now the standard structure returns messages.
              print("--- Warning: Agent executor returned string output, expected AIMessage. ---")
              agent_response_message = AIMessage(content=output_content, tool_calls=[]) # Fallback, likely wrong
         else:
              print(f"--- Error: Unexpected agent executor output format: {response} ---")
              return {"final_summary": "Internal Error: Unexpected response from summarizer agent."}

    elif isinstance(response, AIMessage):
         # If the agent runnable itself was invoked directly
         agent_response_message = response
    else:
          print(f"--- Error: Unexpected agent executor response type: {type(response)} ---")
          return {"final_summary": "Internal Error: Unexpected response from summarizer agent."}


    # Ensure we have an AIMessage to return
    if not isinstance(agent_response_message, AIMessage):
         print(f"--- Error: Could not extract valid AIMessage from agent response. ---")
         return {"final_summary": "Internal Error: Failed to get valid response from summarizer agent."}


    # Return the agent's response to update the state's messages
    return {"messages": [agent_response_message]}


# 5. Node to execute the summarize_content_tool
# We need to instantiate ToolNode here or pass it via config
# summarize_tool_node = ToolNode([summarize_content_tool]) # Instantiate inside build_graph

# 6. Node to call the Formatter Agent (final step)
def call_formatter_agent(state: AgentState, formatter_llm) -> dict:
    """Invokes the formatter LLM one last time to format the raw summary."""
    print("--- Calling Formatter Agent Node ---")

    # Find the raw summary from the last ToolMessage
    raw_summary = None
    error_summary = None
    last_message = state['messages'][-1] if state['messages'] else None

    if isinstance(last_message, ToolMessage) and last_message.name == summarize_content_tool.name:
         try:
             # Safer parsing of the tool output string
             # Try ast.literal_eval first, then potentially json.loads if needed
             tool_output_content = last_message.content
             if isinstance(tool_output_content, str):
                 try:
                     tool_output = ast.literal_eval(tool_output_content)
                 except (SyntaxError, ValueError):
                     # Fallback or error if not a Python literal
                     print("--- Warning: Summary tool output not a Python literal, trying JSON. ---")
                     try:
                         tool_output = json.loads(tool_output_content)
                     except json.JSONDecodeError:
                          raise ValueError("Tool output is not a valid Python literal or JSON string.")
             elif isinstance(tool_output_content, dict): # If already a dict
                  tool_output = tool_output_content
             else:
                  raise TypeError("Tool output content is not a string or dictionary.")


             if isinstance(tool_output, dict):
                 if "summary" in tool_output:
                     raw_summary = tool_output['summary']
                     # Ensure raw_summary is a string
                     if not isinstance(raw_summary, str):
                         raw_summary = str(raw_summary) # Coerce to string
                     state['raw_summary'] = raw_summary # Store raw summary
                     print(f"--- Raw summary obtained. ---")
                 elif "error" in tool_output:
                     error_summary = tool_output['error']
                     print(f"--- Summarization tool reported error: {error_summary} ---")
                     state['final_summary'] = f"Summarization failed: {error_summary}"
                     # Return final error message from AI
                     return {"messages": [AIMessage(content=state['final_summary'], tool_calls=[])]}
                 else:
                      raise ValueError("Summary tool output dict missing 'summary' or 'error'.")
             else:
                 raise TypeError("Parsed tool output is not a dictionary.")

         except Exception as e:
             error_summary = f"Error parsing summary tool result: {e}"
             print(f"--- {error_summary} ---")
             state['final_summary'] = error_summary
             return {"messages": [AIMessage(content=state['final_summary'], tool_calls=[])]}
    else:
        # This case might occur if the summarizer agent itself failed or didn't call the tool
        print("--- Warning/Error: Expected last message to be summary ToolMessage. Trying to format anyway if raw summary exists. ---")
        if state.get('raw_summary'):
             raw_summary = state['raw_summary']
             print("--- Found raw summary in state, proceeding with formatting. ---")
        elif state.get('final_summary'): # If an error was already set
              print("--- An error occurred before formatting. Returning existing error. ---")
              return {"messages": [AIMessage(content=state['final_summary'], tool_calls=[])]}
        else:
              state['final_summary'] = "Internal Error: Could not find raw summary to format."
              return {"messages": [AIMessage(content=state['final_summary'], tool_calls=[])]}

    # Ensure we have a string summary to format
    if not raw_summary or not isinstance(raw_summary, str):
         print("--- No valid raw summary found or is not a string, cannot format. ---")
         if not state.get('final_summary'): # Set final summary if not already set
              state['final_summary'] = "Failed to generate valid summary, cannot format."
         return {"messages": [AIMessage(content=state['final_summary'], tool_calls=[])]}


    # Construct the prompt for the final formatting step
    formatting_prompt = HumanMessage(
        content=f"Here is the generated summary:\n```\n{raw_summary}\n```\n\n"
                f"Now, please format this summary based on the original request: '{state.get('user_input', '')}'. "
                f"If no specific format was requested, just present the summary clearly and concisely. "
                f"Provide *only* the final, formatted summary in your response, without any preamble like 'Here is the formatted summary:'."
    )

    # Add this instruction to the messages history for the formatter LLM
    # Pass *only* the relevant history + new prompt to the formatter LLM
    # Avoid passing tool call/result messages if they confuse the formatter
    formatter_messages = [
        SystemMessage(content="You are a helpful assistant that reformats provided text based on user instructions."),
        HumanMessage(content=f"Original request: {state.get('user_input', '')}"), # Give context
        HumanMessage(content=f"Summary to format:\n{raw_summary}"),
        formatting_prompt # The specific instruction
    ]


    # Invoke the formatter LLM (passed as argument)
    final_response = formatter_llm.invoke(formatter_messages)
    print(f"--- Formatter Agent LLM Response: {final_response.content} ---")

    final_summary_content = final_response.content if isinstance(final_response.content, str) else "Error: Formatter produced invalid content."

    # Update the state with the final summary
    # Also return the final AI message to be added to the history
    return {
        "final_summary": final_summary_content,
        "messages": [AIMessage(content=final_summary_content, tool_calls=[])] # Final message from AI
    }

# --- Graph Definition ---
def build_graph(model_name: str = "gemini-pro"):
    """Builds the LangGraph agent graph."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Error: GOOGLE_API_KEY not found.")

    # LLM for Agent decisions (tool calling)
    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0, convert_system_message_to_human=True)
    agent_llm_with_tools = llm.bind_tools(tools)
    agent_executor = agent_llm_with_tools

    # Separate LLM instance for formatting
    formatter_llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.1, convert_system_message_to_human=True)

    # Tool node for executing the summarize tool
    summarize_tool_node = ToolNode([summarize_content_tool])

    # Define the graph workflow
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("initial_router", route_initial_request)
    workflow.add_node("fetcher", call_fetcher)
    workflow.add_node("decide_chain_type", decide_chain_type_node)
    workflow.add_node("agent_summarizer", lambda state: call_summarizer_agent(state, agent_executor))
    workflow.add_node("summarize_tool_exec", summarize_tool_node)
    workflow.add_node("agent_formatter", lambda state: call_formatter_agent(state, formatter_llm))

    # Define edges
    workflow.set_entry_point("initial_router")

    # --- MODIFIED CONDITIONAL EDGE with DEBUG ---
    # Define a separate function for the conditional logic to add prints
    def decide_after_initial_route(state: AgentState) -> str:
        """
        Determines the next step after the initial routing based on the presence of a URL in the state.
        """
        print("--- Conditional Edge: Running 'decide_after_initial_route' ---")
        url_in_state = state.get('url')
        print(f"    State check: url = {url_in_state}")
        if url_in_state:
            decision = "fetch"
            print(f"    Decision -> '{decision}'")
            return decision
        else:
            decision = "end_no_url"
            print(f"    Decision -> '{decision}'")
            return decision

    # Branch after initial routing using the debug function
    workflow.add_conditional_edges(
        "initial_router",
        decide_after_initial_route, # Use the function with prints
        {
            "fetch": "fetcher",
            "end_no_url": END
        }
    )
    # --- END MODIFICATION ---


    # After fetching
    workflow.add_conditional_edges(
        "fetcher",
        lambda state: "error" if state.get('fetch_error') else "decide",
        {"error": END, "decide": "decide_chain_type"}
    )

    # After deciding chain type
    workflow.add_conditional_edges(
        "decide_chain_type",
        lambda state: "error" if state.get('fetch_error') else "summarize",
        {"error": END, "summarize": "agent_summarizer"}
    )

    # After agent tries to summarize
    workflow.add_conditional_edges(
        "agent_summarizer",
        tools_condition,
        {
            "tools": "summarize_tool_exec",
             END: END
        }
    )

    # After summarize tool execution
    workflow.add_edge("summarize_tool_exec", "agent_formatter")

    # After formatting
    workflow.add_edge("agent_formatter", END)

    # Compile the graph
    graph = workflow.compile()
    print("--- LangGraph Compiled ---")
    return graph

    """Builds the LangGraph agent graph."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Error: GOOGLE_API_KEY not found.")

    # LLM for Agent decisions (tool calling)
    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0, convert_system_message_to_human=True)
    # Bind tools to the LLM to create the agent logic part
    agent_llm_with_tools = llm.bind_tools(tools)

    # Define the agent executor runnable.
    # This will take the state's messages and invoke the LLM with tools.
    agent_executor = agent_llm_with_tools # Direct invocation for simplicity here

    # Separate LLM instance for formatting (can use same model)
    # Ensure it's configured appropriately if needed (e.g., temperature)
    formatter_llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.1, convert_system_message_to_human=True)

    # Tool node for executing the summarize tool
    summarize_tool_node = ToolNode([summarize_content_tool])

    # Define the graph workflow
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("initial_router", route_initial_request)
    workflow.add_node("fetcher", call_fetcher)
    workflow.add_node("decide_chain_type", decide_chain_type_node)
    # Pass the agent_executor to the node function
    workflow.add_node("agent_summarizer", lambda state: call_summarizer_agent(state, agent_executor))
    workflow.add_node("summarize_tool_exec", summarize_tool_node)
    # Pass the formatter_llm instance to the node function
    workflow.add_node("agent_formatter", lambda state: call_formatter_agent(state, formatter_llm))

    # Define edges
    workflow.set_entry_point("initial_router")

    # Branch after initial routing
    workflow.add_conditional_edges(
        "initial_router",
        # Function to decide the next step based on state['url'] being set or not
        lambda state: "fetch" if state.get('url') else "end_no_url",
        {
            "fetch": "fetcher",
            "end_no_url": END # END is a special node name indicating termination
        }
    )
    # Removed extra parenthesis here

    # After fetching
    workflow.add_conditional_edges(
        "fetcher",
        # Route based on presence of fetch_error in the state *after* fetcher runs
        lambda state: "error" if state.get('fetch_error') else "decide",
        {"error": END, "decide": "decide_chain_type"}
    )

    # After deciding chain type
    workflow.add_conditional_edges(
        "decide_chain_type",
         # Check error again (might be set if fetched text was empty)
        lambda state: "error" if state.get('fetch_error') else "summarize",
        {"error": END, "summarize": "agent_summarizer"}
    )

    # After agent tries to summarize (calls tool or responds directly)
    workflow.add_conditional_edges(
        "agent_summarizer",
        # Use prebuilt tools_condition to check if the *last message* in the state contains tool calls
        tools_condition,
        {
            # If tools_condition is True (tool calls present), execute the summarize tool
            "tools": "summarize_tool_exec",
             # If tools_condition is False (no tool calls, e.g., error message), end.
             END: END
        }
    )

    # After summarize tool execution -> Go to format the result
    workflow.add_edge("summarize_tool_exec", "agent_formatter")

    # After formatting -> END (agent_formatter produces the final response)
    workflow.add_edge("agent_formatter", END)

    # Compile the graph
    graph = workflow.compile()
    print("--- LangGraph Compiled ---")
    return graph

# --- Execution Function ---
def run_agent(user_input: str, model_name: str = "gemini-pro"):
    """
    Runs the summarization agent for a given user request (URL + optional format).

    Args:
        user_input (str): The user's full request string.
        model_name (str): The Gemini model to use for the agent.

    Returns:
        str: The final formatted summary or an error message.
    """
    graph = None # Initialize graph to None
    try:
        graph = build_graph(model_name)
        # Initial state
        initial_state = AgentState(
            messages=[HumanMessage(content=user_input)],
            user_input=user_input,
            url=None, # Explicitly set initial optional fields to None
            fetched_text=None,
            fetch_error=None,
            determined_chain_type=None,
            raw_summary=None,
            final_summary=None
        )

        print(f"\n--- Running Agent for Input: '{user_input}' ---")

        final_state = None # Initialize final_state

        # Use stream for better debugging visibility
        for step in graph.stream(initial_state):
            node_name = list(step.keys())[0]
            node_output = step[node_name] # Get the output dict directly
            print(f"--- Output from node '{node_name}': ---")
            # Print state changes or relevant info. Be careful printing large text fields.
            if node_output:
                 print(f"  State Updates: { {k: (v[:100] + '...' if isinstance(v, str) and len(v)>100 else v) for k,v in node_output.items() if k != 'messages'} }") # Print updates nicely
                 if 'messages' in node_output and node_output['messages']:
                     print(f"  Last Message Added: {node_output['messages'][-1].type} - {node_output['messages'][-1].content[:150]}...")
                 if node_output.get('final_summary'):
                      print(f"  Final Summary Set/Updated.")
            else:
                 print("  (No updates returned by node)")
            # Keep track of the last complete state object implicitly handled by stream/invoke
            # final_state = step # Keep the latest step info if needed outside loop

        # After streaming, invoke again to get the *final* accumulated state
        # Streaming itself might not return the *very* final state in a single last event easily
        print("\n--- Invoking graph to get final state ---")
        final_state = graph.invoke(initial_state)


        print("\n--- Agent Execution Finished ---")

        if final_state is None:
             print("--- Error: Final state is None after execution. ---")
             return "Agent finished unexpectedly without a final state."

        # The final summary should be in the state
        final_result = final_state.get('final_summary')
        if final_result is None:
             # Fallback if final_summary wasn't set (e.g., unexpected graph termination)
              print("--- Warning: final_summary key not found in final state or is None. ---")
              # Check if the last message contains the answer
              last_message = final_state.get('messages', [])[-1] if final_state.get('messages') else None
              if isinstance(last_message, AIMessage):
                  final_result = last_message.content
              else:
                  final_result = "Agent finished, but the final summary is missing."

        print(f"--- Final Result: {final_result} ---")
        return final_result if isinstance(final_result, str) else str(final_result)


    except Exception as e:
        print(f"--- Agent Execution Error: {e} ---")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        # Try to return error from state if available
        if graph and graph.nodes: # Check if graph was built
            try:
                 # Attempt to get the state if an intermediate error occurred
                 # This might not always work depending on where the error happened
                 current_state = graph.get_state(initial_state) # Needs configuration passed if used
                 error = current_state.get('fetch_error') or current_state.get('final_summary')
                 if error:
                     return f"An error occurred during agent execution: {error}"
            except Exception as state_err:
                 print(f"--- Error getting state after main exception: {state_err} ---")
                 pass # Fall through to generic error

        return f"An critical error occurred during agent execution: {str(e)}"


# --- Simple Command-Line Test (Optional) ---
if __name__ == "__main__":
    # Test cases
    # test_input = "Can you summarize https://lilianweng.github.io/posts/2023-06-23-agent/ in bullet points?"
    test_input = "Summarize https://blog.google/technology/ai/google-gemini-ai/ as a short paragraph of 2-3 sentences."
    # test_input = "Tell me about this page: https://www.invalid-url-that-will-fail.xyz" # Test fetch error
    # test_input = "Hi there" # Test no URL case
    # test_input = "Summarize https://httpbin.org/html" # Test with simple valid HTML page
    # test_input = "Summarize https://httpbin.org/robots.txt" # Test with plain text page

    # Use a model known to be good with tool calling and following instructions
    # Flash is fast, but might not be as robust for complex formatting as Pro
    # model_to_use="gemini-1.5-flash"
    model_to_use="gemini-pro"

    result = run_agent(test_input, model_name=model_to_use)
    print("\n=========== AGENT FINAL RESULT =============")
    print(result)
    print("===========================================\n")