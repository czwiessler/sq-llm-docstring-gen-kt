import re
import os
from openai import OpenAI
from dotenv import load_dotenv
import shutil


def import_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()


def extract_definitions_from_code(code):
    header_regex = re.compile(r'^(\s*)(class|def|async\s+def)\s+')
    lines = code.splitlines()

    definitions = []
    for idx, line in enumerate(lines):
        match = header_regex.match(line)
        if match:
            indent = len(match.group(1))
            definitions.append((idx, line.rstrip(), indent))

    return definitions


def group_class_methods(definitions):
    class_groups = {}
    current_class = None
    for idx, header, indent in definitions:
        if header.startswith("class"):
            current_class = header
            class_groups[current_class] = [header]
        elif current_class and indent > 0:
            class_groups[current_class].append(header)
        else:
            current_class = None
    return class_groups


def extract_code_blocks(lines, definitions, class_groups):
    blocks = {}

    for i, (idx, header, indent) in enumerate(definitions):
        start_idx = idx
        end_idx = len(lines)

        # Determine the end index of the current block
        if i + 1 < len(definitions):
            end_idx = definitions[i + 1][0]

        block_lines = []
        inside_block = False

        # Finde heraus, ob das aktuelle `header` zu einer Klasse oder einer Methode gehört
        class_header = None
        for class_name, methods in class_groups.items():
            if header in methods:
                class_header = class_name
                break  # Stoppe, sobald die Klasse gefunden wurde



        for j in range(start_idx, end_idx):
            line = lines[j]
            current_indent = len(re.match(r'^(\s*)', line).group(1))

            if j == start_idx:  # Always include the header line
                block_lines.append(line)
                inside_block = True
            elif inside_block:
                # Include lines with greater or equal indentation or blank lines
                if line.strip() == '' or current_indent >= indent:
                    block_lines.append(line)
                else:
                    break
        # Falls die Methode zu einer Klasse gehört, füge die Klassendefinition + alle anderen Methoden-Header ein
        if class_header:
            block_lines.append('FYI: Other headers of this class:\n')
            for method in class_groups[class_header]:
                if method != header:  # Vermeide die aktuell betrachtete Methode/Klasse
                    block_lines.append(method + '\n')
        blocks[header] = ''.join(block_lines)

    return blocks


def save_blocks(blocks, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, (header, content) in enumerate(blocks.items(), 1):
        filename = f"{i}_" + re.sub(r'\W+', '_', header.strip()) + ".py"
        with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as file:
            file.write(content)


def remove_code_fences(text):
    return text.strip('`').strip()


def generate_docstrings_for_chunk(client, model, filename, chunk_code, system_prompt, user_prompt):
    full_prompt = user_prompt + chunk_code
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt},
        ],
        model=model,
    )
    chunk_generated = response.choices[0].message.content
    chunk_generated_cleaned = remove_code_fences(chunk_generated)
    return chunk_generated_cleaned


def get_docstrings(output_dir, docstring_dir):
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY", "keyhere")
    client = OpenAI(api_key=api_key)
    system_prompt = "You are a helpful assistant for analyzing Python code and generating inline docstrings."

    model = "gpt-4o"

    os.makedirs(docstring_dir, exist_ok=True)

    for filename in os.listdir(output_dir):
        if filename.endswith(".py"):
            file_path = os.path.join(output_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                code = file.read()

            function_user_prompt = """
                IMPORTANT: Give me just the docstring and do not use triple quotes or ``` or ''' in the response.
                Generate a comprehensive Python docstring for the following function. 
                The docstring should include:

                1. A clear and concise description of the purpose.
                2. A detailed explanation of all parameters with their data types.
                3. A description of the return value(s) with their data types.
                4. Information about possible exceptions (errors) the function may raise.
                
                Formatting rules:  
                - Use exactly 1 tab (4 spaces) for indentation.  
                - Ensure that `Args`, `Returns`, and `Raises` are NEVER indented.  
                
                Example for a function style:
                def add(a: int, b: int) -> int:
                    \"""
                    Adds two integers and returns the result.

                    Args:
                        a (int): The first integer.
                        b (int): The second integer.

                    Returns:
                        int: The sum of `a` and `b`.
                    \"""

                IMPORTANT: 
                - Do not include the function definition or Python at the beginning.
                """
            class_user_prompt = """
                IMPORTANT: Give me just the docstring and do not use triple quotes or ``` or ''' in the response.
                Generate a comprehensive Python docstring for the following class. 
                The docstring should include:

                1. A clear and concise description of the purpose.
                2. A detailed explanation of all parameters with their data types.
                4. Information about possible exceptions (errors) the class may raise.

                a. Provide a general description of the class purpose.
                b. Include a description of the class attributes (if any) with their data types.
                c. For each method in the class, include a brief description (do not generate full docstrings for methods).

                Formatting rules:  
                - Use exactly 1 tab (4 spaces) for indentation.
                - Do not indent sections like `Attributes` or `Methods` etc.
                
                Example for a class style:
                class Calculator:
                    \"""
                    A simple calculator class for basic arithmetic operations.

                    Attributes:
                        precision (int): The number of decimal places for results.

                    Methods:
                        add(a: float, b: float) -> float:
                            Adds two numbers and returns the result.
                        subtract(a: float, b: float) -> float:
                            Subtracts the second number from the first.
                    \"""

                IMPORTANT: 
                - Do not include the class definition or Python at the beginning.
                """
            print("Processing ", filename)
            if "def" in filename:
                docstring = generate_docstrings_for_chunk(client, model, filename, code, system_prompt, function_user_prompt)
            elif "class" in filename:
                docstring = generate_docstrings_for_chunk(client, model, filename, code, system_prompt, class_user_prompt)
            else:
                print("Unknown type in ", filename)
                continue

            # if the docstring contains triple quotes, try again 2 more times
            if '"""' in docstring or "'''" in docstring or '```' in docstring or "    Args:" in docstring:
                stronger_function_user_prompt = """
                    IMPORTANT: Give me just the docstring and do not use triple quotes in the response.
                    Generate a comprehensive Python docstring for the following function. 
                    The docstring should include:

                    1. A clear and concise description of the purpose.
                    2. A detailed explanation of all parameters with their data types.
                    3. A description of the return value(s) with their data types.
                    4. Information about possible exceptions (errors) the function may raise.
                
                    Formatting rules:  
                    - Use exactly 1 tab (4 spaces) for indentation.  
                    - Ensure that `Args`, `Returns`, and `Raises` are NEVER indented.  
                    
                    Example for a function style:
                    def add(a: int, b: int) -> int:
                        \"""
                        Adds two integers and returns the result.

                        Args:
                            a (int): The first integer.
                            b (int): The second integer.

                        Returns:
                            int: The sum of `a` and `b`.
                        \"""

                    IMPORTANT: 
                    - Provide only the docstring and do not use triple quotes or ` or ' in the response. 
                    - Do not include the function definition or Python at the beginning.
                    """
                stronger_class_user_prompt = """
                    IMPORTANT: Give me just the docstring and do not use triple quotes in the response.
                    Generate a comprehensive Python docstring for the following class. 
                    The docstring should include:

                    1. A clear and concise description of the purpose.
                    2. A detailed explanation of all parameters with their data types.
                    4. Information about possible exceptions (errors) the class may raise.

                    a. Provide a general description of the class purpose.
                    b. Include a description of the class attributes (if any) with their data types.
                    c. For each method in the class, include a brief description (do not generate full docstrings for methods).

                    Formatting rules:  
                    - Use exactly 1 tab (4 spaces) for indentation.
                    - Do not indent sections like `Attributes` or `Methods` etc.
                    
                    Example for a class style:
                    class Calculator:
                        \"""
                        A simple calculator class for basic arithmetic operations.

                        Attributes:
                            precision (int): The number of decimal places for results.

                        Methods:
                            add(a: float, b: float) -> float:
                                Adds two numbers and returns the result.
                            subtract(a: float, b: float) -> float:
                                Subtracts the second number from the first.
                        \"""

                    IMPORTANT: 
                    - Provide only the docstring and do not use triple quotes or ` or ' in the response. 
                    - Do not include the class definition or Python at the beginning.
                    """
                print("Retrying 1 for ", filename)
                if "def" in filename:
                    docstring = generate_docstrings_for_chunk(client, model, filename, code, system_prompt, stronger_function_user_prompt)
                elif "class" in filename:
                    docstring = generate_docstrings_for_chunk(client, model, filename, code, system_prompt, stronger_class_user_prompt)
            if '"""' in docstring or "'''" in docstring or '```' in docstring or  "    Args:" in docstring:
                strongest_function_user_prompt = """
                    MOST IMPORTANT: Give me just the docstring and do not use triple quotes in the response.
                    DO NOT USE TRIPLE QUOTES IN THE RESPONSE.
                    Generate a comprehensive Python docstring for the following function. 
                    The docstring should include:
    
                    1. A clear and concise description of the purpose.
                    2. A detailed explanation of all parameters with their data types.
                    3. A description of the return value(s) with their data types.
                    4. Information about possible exceptions (errors) the function may raise.
                    
                    Formatting rules:  
                    - Use exactly 1 tab (4 spaces) for indentation.  
                    - Ensure that `Args`, `Returns`, and `Raises` are NEVER indented.  
                    
                    Example for a function style:
                    def add(a: int, b: int) -> int:
                        \"""
                        Adds two integers and returns the result.
    
                        Args:
                            a (int): The first integer.
                            b (int): The second integer.
    
                        Returns:
                            int: The sum of `a` and `b`.
                        \"""
    
                    IMPORTANT: 
                    - Provide only the docstring and do not use triple quotes or ` or ' in the response. 
                    - Do not include the function definition or Python at the beginning.
                    """
                strongest_class_user_prompt = """
                    MOST IMPORTANT: Give me just the docstring and do not use triple quotes in the response.
                    DO NOT USE TRIPLE QUOTES IN THE RESPONSE.
                    Generate a comprehensive Python docstring for the following class. 
                    The docstring should include:
    
                    1. A clear and concise description of the purpose.
                    2. A detailed explanation of all parameters with their data types.
                    4. Information about possible exceptions (errors) the class may raise.
    
                    a. Provide a general description of the class purpose.
                    b. Include a description of the class attributes (if any) with their data types.
                    c. For each method in the class, include a brief description (do not generate full docstrings for methods).
    
                    Formatting rules:  
                    - Use exactly 1 tab (4 spaces) for indentation.
                    - Do not indent sections like `Attributes` or `Methods` etc.
                    
                    Example for a class style:
                    class Calculator:
                        \"""
                        A simple calculator class for basic arithmetic operations.
    
                        Attributes:
                            precision (int): The number of decimal places for results.
    
                        Methods:
                            add(a: float, b: float) -> float:
                                Adds two numbers and returns the result.
                            subtract(a: float, b: float) -> float:
                                Subtracts the second number from the first.
                        \"""
    
                    IMPORTANT: 
                    - Provide only the docstring and do not use triple quotes or ` or ' in the response. 
                    - Do not include the class definition or Python at the beginning.
                    """
                print("Retrying 2 for ", filename)
                if "def" in filename:
                    docstring = generate_docstrings_for_chunk(client, model, filename, code, system_prompt,
                                                              strongest_function_user_prompt)
                elif "class" in filename:
                    docstring = generate_docstrings_for_chunk(client, model, filename, code, system_prompt,
                                                              strongest_class_user_prompt)
                # if there is still a triple quote, print the filename
                if '"""' in docstring or "'''" in docstring or '```' or  "    Args:" in docstring:
                    print("Triple quote or wrong indentations in ", filename, "Deleting docstring...")
                    # clear the docstring
                    docstring = ""

            # Speichere den Docstring als reine Textdatei
            docstring_filename = os.path.splitext(filename)[0] + ".txt"
            with open(os.path.join(docstring_dir, docstring_filename), 'w', encoding='utf-8') as doc_file:
                doc_file.write(docstring)


def clean_docstrings(docstring_dir):
    for filename in os.listdir(docstring_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(docstring_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                docstring = file.read()

            # Entferne Blöcke, die mit Args:, Returns: oder Raises: beginnen und None enthalten
            cleaned_docstring = re.sub(
                r'\n\s*(Args|Returns|Raises):\s*None(?:\n\s*:[^\n]*)*\n?',
                '\n',
                docstring,
                flags=re.MULTILINE
            )

            # Falls nach "Args:" nichts mehr Relevantes steht, die leere Zeile danach erhalten
            cleaned_docstring = re.sub(
                r'\n\s*(Args|Returns|Raises):\s*\n\s*\n?',
                '\n',
                cleaned_docstring,
                flags=re.MULTILINE
            )

            # Speichere die bereinigte Version mit korrektem Zeilenabstand
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(cleaned_docstring.strip() + '\n')  # Abschließendes Newline für Konsistenz



def remove_docstrings_from_original_code(code: str) -> str:
    """
    Entfernt alle Docstrings aus dem Code, fügt aber (optional) den ursprünglichen
    Modul-Docstring wieder vorne an, wenn es einen gab.
    """
    module_docstring_pattern = r'\A\s*("""|\'\'\')((?:.|\n)*?)\1'
    module_match = re.match(module_docstring_pattern, code, flags=re.DOTALL)
    module_docstring = module_match.group(0) if module_match else ''

    all_docstring_pattern = r'("""|\'\'\')((?:.|\n)*?)\1'
    cleaned_code = re.sub(all_docstring_pattern, '', code, flags=re.DOTALL)

    if module_docstring:
        cleaned_code = module_docstring + '\n' + cleaned_code.lstrip()

    return cleaned_code

def insert_docstrings(original_code, docstring_dir):
    """
    Iteriert durch den Originalcode und fügt Docstrings aus `docstring_dir` in der Reihenfolge der Dateien ein.
    """
    lines = original_code.splitlines(True)

    # Lade alle Docstring-Dateien in sortierter Reihenfolge
    docstring_files = sorted(os.listdir(docstring_dir), key=lambda x: int(x.split("_")[0]))
    docstring_index = 0  # Verfolgt, welcher Docstring als nächstes eingefügt wird

    for i, line in enumerate(lines):
        if line.strip().startswith(("def ", "async def ", "class ")) and docstring_index < len(docstring_files):
            definition = line.strip()
            j = i

            while not definition.endswith(":"):
                j += 1
                definition += lines[j].strip()

            # Lade den nächsten Docstring (in der gespeicherten Reihenfolge)
            docstring_path = os.path.join(docstring_dir, docstring_files[docstring_index])
            with open(docstring_path, "r") as file:
                docstring_content = file.read()
            docstring_index += 1  # Zum nächsten Docstring gehen

            # Berechne die Einrückung für den Docstring
            indent = len(line) - len(line.lstrip())
            docstring_indent = " " * (indent + 4)

            # Formatiere den Docstring mit passender Einrückung
            formatted_docstring = (
                f'{docstring_indent}"""\n'
                + "\n".join(docstring_indent + l for l in docstring_content.splitlines())
                + f'\n{docstring_indent}"""\n'
            )

            # Füge den Docstring nach der Funktions-/Klassendefinition ein
            lines[j + 1:j + 1] = formatted_docstring.splitlines(True)

    return "".join(lines)


def delete_docstring_and_blocks_dir(docstring_dir, output_dir):
    for dir_path in [docstring_dir, output_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"Deleted directory: {dir_path}")
        else:
            print(f"Directory does not exist: {dir_path}")


if __name__ == "__main__":
    file_path = "../downloaded_files/pyro-ppl/integrate.py"
    #file_path = "extract.py"
    #file_path = "../examples/run_C/test_streaming_original.py"
    output_dir = "extracted_blocks"
    docstring_dir = "docstrings"

    file_lines = import_file(file_path)
    code = "".join(file_lines)
    definitions = extract_definitions_from_code(code)
    class_groups = group_class_methods(definitions)
    blocks = extract_code_blocks(file_lines, definitions, class_groups)
    save_blocks(blocks, output_dir)
    get_docstrings(output_dir, docstring_dir)

    #clean_docstrings(docstring_dir)

    clean_code = remove_docstrings_from_original_code(code)

    commented_code = insert_docstrings(clean_code, docstring_dir)

    delete_docstring_and_blocks_dir(docstring_dir, output_dir)

    with open("commented_code.py", "w", encoding="utf-8") as output_file:
        output_file.write(commented_code)

    print(f"Extracted {len(blocks)} blocks to '{output_dir}' and generated docstrings in 'docstrings'.")
