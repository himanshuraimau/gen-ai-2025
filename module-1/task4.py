# Standard library imports
import os
import sys

# Third-party imports
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import trim_messages

# Setup path for local imports
script_dir = os.getcwd()
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Local imports
from helper import OPENAI_API_KEY, get_api_key, validate_api_keys, GEMINI_API_KEY, GROQ_API_KEY

def init_model():
    """Initialize the chat model with OpenAI."""
    return init_chat_model(
        "gpt-3.5-turbo",
        model_provider="openai"
    )

def setup_workflow():
    """Setup the conversation workflow with state management."""
    # Define state schema with both messages and language
    class ConversationState(MessagesState):
        language: str = "English"  # Default language
        
    workflow = StateGraph(state_schema=ConversationState)
    
    # Configure message trimmer
    trimmer = trim_messages(
        max_tokens=65,
        strategy="last",
        token_counter=model,
        include_system=True,
        allow_partial=False,
        start_on="human",
    )
    
    def call_model(state):
        """Process messages through the model."""
        trimmed_messages = trimmer.invoke(state["messages"])
        prompt = prompt_template.invoke({
            "messages": trimmed_messages,
            "language": state.get("language", "English")  # Use get with default
        })
        response = model.invoke(prompt)
        return {"messages": [response]}
    
    workflow.add_edge(START, "model") # adding the edge to the graph
    workflow.add_node("model", call_model) # it is adding the node to the graph
    return workflow

# Initialize the model
model = init_model()

# Setup example messages for testing
messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

# Setup workflow and memory
workflow = setup_workflow()
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Configure prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You talk like a pirate. Answer all questions to the best of your ability."),
    MessagesPlaceholder(variable_name="messages"),
])

# Test configuration
config = {"configurable": {"thread_id": "abc345"}}
language = "English"

# Test queries
def run_query(query, prev_messages=None):
    """Run a query through the conversation workflow."""
    input_messages = [HumanMessage(query)]
    if prev_messages:
        input_messages = prev_messages + input_messages
    
    # Initialize state with both messages and language
    state = {
        "messages": input_messages,
        "language": language
    }
    return app.invoke(state, config)

# Example usage
if __name__ == "__main__":
    # First query with context
    output = run_query("Hi! I'm Jim.", messages)
    output["messages"][-1].pretty_print()
    
    # Second query without context
    output = run_query("What is my name?")
    output["messages"][-1].pretty_print()