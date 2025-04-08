import os
import sys
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
import argparse

# Add the parent directory to the system path for module imports
script_dir = os.getcwd()
parent_dir = os.path.dirname(script_dir)

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import API keys and validation functions from helper module
from helper import OPENAI_API_KEY, get_api_key, validate_api_keys,GEMINI_API_KEY,GROQ_API_KEY


# # Initialize the chat model (GPT-3.5 Turbo)
# model = init_chat_model(
#     "gpt-3.5-turbo",
#     model_provider="openai"
# )

model = init_chat_model("llama3-8b-8192", model_provider="groq")

# Zero-shot prompting example
def zero_shot_prompt():
    """
    Demonstrates zero-shot prompting where the model is given a task without any examples.
    """
    messages = [
        SystemMessage(
            content="You are a helpful assistant."
        ),
        HumanMessage(
            content="Translate the following English text to French: 'Hello, how are you?'"
        )
    ]
    print("Zero-shot Prompt:", model.invoke(messages))

# One-shot prompting example
def one_shot_prompt():
    """
    Demonstrates one-shot prompting by providing one example to the model.
    The model learns from this example and applies the same logic to new inputs.
    """
    messages = [
        SystemMessage(
            content="You are a helpful assistant."
        ),
        HumanMessage(
            content="Translate the following English text to French: 'Hello, how are you?'"
        ),
        AIMessage(
            content="Bonjour, comment ça va ?"
        ),
        HumanMessage(
            content="Translate the following English text to French: 'Goodbye!'"
        )
    ]

    print("One-shot Prompt:", model.invoke(messages))

# Few-shot prompting example
def few_shot_prompt():
    """
    Demonstrates few-shot prompting by providing multiple examples to the model.
    This helps the model to better understand the task and provide more accurate responses.
    """
    messages = [
        SystemMessage(
            content="You are a helpful assistant that translates English to French."
        ),
        HumanMessage(
            content="Translate the following English text to French: 'Hello, how are you?'"
        ),
        AIMessage(
            content="Bonjour, comment ça va ?"
        ),
        HumanMessage(
            content="Translate the following English text to French: 'Goodbye!'"
        ),
        AIMessage(
            content="Au revoir !"
        ),
        HumanMessage(
            content="Translate the following English text to French: 'What is your name?'"
        ),
        AIMessage(
            content="Comment vous appelez-vous ?"
        ),
        HumanMessage(
            content="Translate the following English text to French: 'I am learning AI.'"
        )
    ]

    print("Few-shot Prompt:", model.invoke(messages).content)

# Chain-of-thought prompting example
def chain_of_thought_prompt():
    """
    Demonstrates chain-of-thought prompting where the model explains its reasoning
    before providing the final answer. This can improve the accuracy and transparency
    of the model's responses.
    """
    messages = [
        SystemMessage(content="You are a helpful assistant that explains how translations work."),
        HumanMessage(content="Translate the following English text to French and explain your reasoning: 'I love programming.'")
    ]
    print("Chain-of-thought Prompt:", model.invoke(messages).content)

# Prompt template system example
def prompt_template_system():
    """
    Demonstrates a prompt template system where prompts are dynamically generated
    based on user inputs. This allows for more flexible and reusable prompts.
    """
    system_template = "Translate the following from English into {language}"

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{text}")]
    )
    prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})
    print("Prompt Template System:", prompt.to_messages())
    print("Prompt Template System:", model.invoke(prompt.to_messages()).content)

def main():
    """
    Main function to parse arguments and call the appropriate prompting function.
    """
    parser = argparse.ArgumentParser(description="Run different prompting examples.")
    parser.add_argument(
        "prompt_type",
        choices=[
            "zero_shot",
            "one_shot",
            "few_shot",
            "chain_of_thought",
            "prompt_template"
        ],
        help="Type of prompt to run."
    )

    args = parser.parse_args()

    if args.prompt_type == "zero_shot":
        zero_shot_prompt()
    elif args.prompt_type == "one_shot":
        one_shot_prompt()
    elif args.prompt_type == "few_shot":
        few_shot_prompt()
    elif args.prompt_type == "chain_of_thought":
        chain_of_thought_prompt()
    elif args.prompt_type == "prompt_template":
        prompt_template_system()
    else:
        print("Invalid prompt type.")

if __name__ == "__main__":
    main()
