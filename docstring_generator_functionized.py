import os
from openai import OpenAI
from dotenv import load_dotenv


#based on DocstringGenerator.py, this script shall include the same functionality, but in a functionized manner

def annotate_script(model: str, python_code: str) -> str:
    """
    Annotates a Python script with inline docstrings using the OpenAI API.

    Args:
        api_key (str): OpenAI API key for authentication.
        model (str): OpenAI model to use for generating docstrings.
        output_path (str): Path to save the annotated script.

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

    return annotated_code
