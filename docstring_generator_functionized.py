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
                           f"Only return the class header, function header, or asynchronous function header and the docstring."
                           f"use triple quotes like these \"\"\" as a docstring wrapper"
                           f"\n\n{original_python_code}",
            },
        ],
        model=model,
    )

    generated_annotations = chat_completion.choices[0].message.content

    #delete code tags
    generated_annotations = generated_annotations.split("\n")
    generated_annotations = [line for line in generated_annotations if not line.strip().startswith("```")]
    generated_annotations = "\n".join(generated_annotations)

    #for easier debugging
    # save the generated docstrings (answer) as a file to target_directory. Before saving, attach "_generated_docstrings" to the filename
    script_name = file_name
    script_name = script_name.replace(".py", "_generated_docstrings.py")
    output_path = os.path.join(target_directory, script_name)
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(generated_annotations)


    # remove original docstrings (if existing) before inserting the "new" ones
    cleaned_python_code = remove_docstrings(original_python_code)

    # for easier debugging
    # save the cleaned code as a file to target_directory. Before saving, attach "_cleaned" to the filename
    script_name = file_name
    script_name = script_name.replace(".py", "_cleaned.py")
    output_path = os.path.join(target_directory, script_name)
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(cleaned_python_code)


    annotated_code = map_annotations(cleaned_python_code, generated_annotations)

    return annotated_code


def map_annotations(input_code: str, annotations: str) -> str:

    # Parse the annotations into a dictionary
    def parse_annotations(annotations: str) -> dict:
        annotation_dict = {}
        current_key = None
        current_value = []

        for annotation_line in annotations.splitlines():
            annotation_line = annotation_line.rstrip() #remove trailing whitespaces
            #check if the line starts with arbitrary number of whitespaces followed by "class ", "def ", or "async def "
            if re.match(r'^\s*(class|def|async def)\b', annotation_line):
            #if annotation_line.startswith("class ") or annotation_line.startswith("def ") or annotation_line.startswith("async def "):
                if current_key: #if there is a current key, save the current value to the dict
                    annotation_dict[current_key] = "\n".join(current_value).rstrip() #join the lines of the current value and remove (ONLY!) trailing whitespaces
                current_key = re.match(r'^\s*(class|def|async def)\s+(\w+)', annotation_line).group(2) #group(2) returns the second group of the match, being the name of the class or function
                current_value = []
            elif current_key is not None:
                current_value.append(annotation_line)

        if current_key:
            annotation_dict[current_key] = "\n".join(current_value).strip()

        #save the dict in an indented and readable form that preserves brackets of the dict structure to examples/dicts
        with open("examples/dicts/annotation_dict.txt", "w", encoding="utf-8") as file:
            file.write(str(annotation_dict))

        import pprint
        pprint.pp(annotation_dict)

        return annotation_dict

    annotation_dict = parse_annotations(annotations)

    # Insert annotations into the code
    def insert_annotations(input_code: str, annotation_dict: dict) -> str:
        annotated_code = []
        code_lines = input_code.splitlines()

        for code_line in code_lines:
            annotated_code.append(code_line)
            match = re.match(r'(class|def|async def)\s+(\w+)', code_line)
            if match:
                class_or_func_name = match.group(2)
                if class_or_func_name in annotation_dict:
                    annotated_code.append(annotation_dict[class_or_func_name])

        return "\n".join(annotated_code)

    return insert_annotations(input_code, annotation_dict)


def remove_docstrings(code: str) -> str:
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
