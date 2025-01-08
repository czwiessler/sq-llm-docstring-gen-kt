import os
from openai import OpenAI
from dotenv import load_dotenv
import re


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


def annotate_by_mapping(model: str, python_code: str) -> str:

    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY", "keyhere")

    # OpenAI client
    client = OpenAI(api_key=api_key)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant for analyzing Python code and generating inline docstrings.",
            },
            {
                "role": "user",
                "content": f"Please generate appropriate inline docstrings for both functions and classes in the following Python code, ensuring the file remains executable."
                           f"Only return the docstrings and the classes, functions, or asynchronous functions to which they belong:"
                           f"\n\n{python_code}",
            },
        ],
        model=model,
    )

    answer = chat_completion.choices[0].message.content

    #delete first and last line of the generated code
    answer = answer.split("\n")
    answer = answer[1:-1]
    answer = "\n".join(answer)

    annotated_code = map_annotations(python_code, answer)

    return annotated_code


def map_annotations(input_code: str, annotations: str) -> str:
    """
    Maps annotations to their corresponding classes or functions in the input Python code.

    Args:
        input_code (str): The Python code to annotate.
        annotations (str): The annotations to insert into the Python code.

    Returns:
        str: Annotated Python code.
    """

    # Parse the annotations into a dictionary
    def parse_annotations(annotations):
        annotation_dict = {}
        current_key = None
        current_value = []

        for line in annotations.splitlines():
            line = line.rstrip()
            if line.startswith("class ") or line.startswith("def ") or line.startswith("async def "):
                if current_key:
                    annotation_dict[current_key] = "\n".join(current_value).strip()
                current_key = re.match(r'(class|def|async def)\s+(\w+)', line).group(2)
                current_value = []
            elif current_key is not None:
                current_value.append(line)

        if current_key:
            annotation_dict[current_key] = "\n".join(current_value).strip()

        return annotation_dict

    annotation_dict = parse_annotations(annotations)

    # Insert annotations into the code
    def insert_annotations(input_code, annotation_dict):
        annotated_code = []
        lines = input_code.splitlines()

        for line in lines:
            annotated_code.append(line)
            match = re.match(r'(class|def|async def)\s+(\w+)', line)
            if match:
                name = match.group(2)
                if name in annotation_dict:
                    annotated_code.append(annotation_dict[name])

        return "\n".join(annotated_code)

    return insert_annotations(input_code, annotation_dict)


if __name__ == "__main__":
    model = "gpt-3.5-turbo"
#   with open("downloaded_files/quantumiracle/ppo_gae_continuous3.py", "r", encoding="utf-8") as file:
#   with open("downloaded_files/wesselb/readme_example8_gp-rnn.py", "r", encoding="utf-8") as file:
    with open("downloaded_files/ChenRocks/training.py", "r", encoding="utf-8") as file:
        python_code = file.read()

    print(annotate_by_mapping(model, python_code))
