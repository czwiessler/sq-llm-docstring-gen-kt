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


def annotate_by_mapping(model: str, original_python_code: str, target_directory: str, file_name: str) -> str:

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
                           f"\n\n{original_python_code}",
            },
        ],
        model=model,
    )

    answer = chat_completion.choices[0].message.content

    #delete code tags
    answer = answer.split("\n")
    answer = [line for line in answer if not line.strip().startswith("```")]
    answer = "\n".join(answer)

    # remove original docstrings (if existing) before inserting the "new" ones
    cleaned_python_code = remove_docstrings(original_python_code)

    # save the cleaned code as a file to target_directory. Before saving, attach "_cleaned" to the filename
    script_name = file_name
    script_name = script_name.replace(".py", "_cleaned.py")
    output_path = os.path.join(target_directory, script_name)
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(cleaned_python_code)


    annotated_code = map_annotations(cleaned_python_code, answer)

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
    def parse_annotations(annotations: str) -> dict:
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
    def insert_annotations(input_code: str, annotation_dict: dict) -> str:
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


def remove_docstrings(code: str) -> str:
    """
    Removes all docstrings from the given Python code string, except for module-level docstrings.

    Args:
        code (str): Python code as a string.

    Returns:
        str: Python code with non-module docstrings removed.
    """
    # Regular expression to match module-level docstrings
    module_docstring_pattern = r'\A\s*("""|\'\'\')((?:.|\n)*?)\1'

    # Check for a module-level docstring
    module_match = re.match(module_docstring_pattern, code, flags=re.DOTALL)
    module_docstring = module_match.group(0) if module_match else ''

    # Regular expression to match all docstrings
    all_docstring_pattern = r'("""|\'\'\')((?:.|\n)*?)\1'

    # Remove all docstrings
    cleaned_code = re.sub(all_docstring_pattern, '', code, flags=re.DOTALL)

    # Reattach the module-level docstring, if it exists
    if module_docstring:
        cleaned_code = module_docstring + '\n' + cleaned_code.lstrip()

    return cleaned_code


if __name__ == "__main__":
    model = "gpt-3.5-turbo"
#   with open("downloaded_files/quantumiracle/ppo_gae_continuous3.py", "r", encoding="utf-8") as file:
#   with open("downloaded_files/wesselb/readme_example8_gp-rnn.py", "r", encoding="utf-8") as file:
#   with open("downloaded_files/ChenRocks/training.py", "r", encoding="utf-8") as file:
#   with open("downloaded_files/streamlit/st_magic.py", "r", encoding="utf-8") as file:
    with open("downloaded_files/IDSIA/stdout_capturing.py", "r", encoding="utf-8") as file:
        python_code = file.read()

    print(annotate_by_mapping(model, python_code))
