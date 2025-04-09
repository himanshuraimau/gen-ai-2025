# Standard library imports
import os
import sys
import streamlit as st

# Third-party imports
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
import json

# Setup path for local imports
script_dir = os.getcwd()
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Local imports
from helper import OPENAI_API_KEY, get_api_key, validate_api_keys, GEMINI_API_KEY, GROQ_API_KEY

# Configure prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You talk like a pirate. Answer all questions to the best of your ability."),
    MessagesPlaceholder(variable_name="messages"),
])

def init_model():
    """Initialize the chat model with OpenAI."""
    return init_chat_model(
        "gpt-3.5-turbo",
        model_provider="openai"
    )

def summarize_older_messages(messages, model):
    """Summarize older messages while keeping recent ones intact."""
    if len(messages) <= 5:  # Keep all if less than 5 messages
        return messages
    
    # Split messages: Keep last 5 complete, summarize the rest
    recent_messages = messages[-5:]
    older_messages = messages[:-5]
    
    # Format older messages for better summarization
    older_text = "\n".join([
        f"{m['role'].capitalize()}: {m['content']}\n" 
        for m in older_messages
    ])
    
    # Create document for summarization
    docs = [Document(page_content=older_text)]
    
    # Create and run summarization chain with specific prompt
    chain = load_summarize_chain(
        model,
        chain_type="stuff",
        prompt_template="""
        Summarize the following conversation while preserving key information:
        {text}
        
        Summary:"""
    )
    
    try:
        summary = chain.run(docs)
        # Create a summary message that clearly indicates it's a summary
        summary_message = {
            "role": "system",
            "content": f"[Previous conversation summary: {summary.strip()}]"
        }
        return [summary_message] + recent_messages
    except Exception as e:
        # Fallback: If summarization fails, return recent messages only
        print(f"Summarization failed: {str(e)}")
        return recent_messages

def setup_workflow():
    """Setup the conversation workflow with state management."""
    # Define state schema with both messages and language
    class ConversationState(MessagesState):
        language: str = "English"  # Default language
        
    workflow = StateGraph(state_schema=MessagesState)
    
    
    def call_model(state: MessagesState):
        prompt = prompt_template.invoke(state)
        response = st.session_state.model.invoke(prompt)
        return {"messages": response}

    
    workflow.add_edge(START, "model") # adding the edge to the graph
    workflow.add_node("model", call_model) # it is adding the node to the graph
    return workflow

def run_query(query, prev_messages=None):
    """Run a query through the conversation workflow."""
    input_messages = [HumanMessage(query)]
    if prev_messages:
        input_messages = prev_messages
    
    # Initialize state with both messages and language
    state = {
        "messages": input_messages,
        "language": "English"
    }
    config = {"configurable": {"thread_id": "abc345"}}
    return st.session_state.app.invoke(state, config)

# Initialize the model and workflow
if 'model' not in st.session_state:
    st.session_state.model = init_model()
    st.session_state.workflow = setup_workflow()
    st.session_state.memory = MemorySaver()
    st.session_state.app = st.session_state.workflow.compile(checkpointer=st.session_state.memory)
    st.session_state.messages = []

# Streamlit UI
st.title("Pirate Chat Bot 🏴‍☠️")

# Add clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("What's on your mind, matey?"):
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    try:
        # Show loading state
        with st.chat_message("assistant"):
            with st.spinner("Thinking like a pirate..."):
                # Process messages with summarization
                processed_history = summarize_older_messages(
                    st.session_state.chat_history, 
                    st.session_state.model
                )
                
                # Convert to LangChain message format
                messages = [
                    HumanMessage(m["content"]) if m["role"] == "user" 
                    else AIMessage(m["content"]) if m["role"] == "assistant"
                    else SystemMessage(content=m["content"])
                    for m in processed_history
                ]
                
                output = run_query(prompt, messages)
                ai_response = output["messages"][-1].content

                # Display AI response
                st.write(ai_response)
        
        # Add AI response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
    except Exception as e:
        st.error(f"Arrrr! Something went wrong: {str(e)}")
        # Remove the failed user message from history
        st.session_state.chat_history.pop()