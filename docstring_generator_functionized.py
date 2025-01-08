import os
from openai import OpenAI
from dotenv import load_dotenv


def annotate_script(model: str, python_code: str) -> str:
    """
    Annotates a Python script with inline docstrings using the OpenAI API.

    Args:
        model (str): OpenAI model to use for generating docstrings.
        python_code (str): Python script to annotate with inline docstrings.

    Returns:
        str: The annotated Python script with inline docstrings.
    """


    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY", "keyhere")

    # OpenAI client
    client = OpenAI(api_key=api_key)

    # Request to the Chat-Completion API
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant for analyzing Python code and generating inline docstrings.",
            },
            {
                "role": "user",
                "content": f"Please add appropriate inline docstrings to the following Python code, ensuring the file remains executable:\n\n{python_code}",
            },
        ],
        model=model,
    )

    # Extract generated content
    annotated_code = chat_completion.choices[0].message.content

    #delete code tags
    annotated_code = annotated_code.split("\n")
    annotated_code = [line for line in annotated_code if not line.strip().startswith("```")]
    annotated_code = "\n".join(annotated_code)


    return annotated_code
