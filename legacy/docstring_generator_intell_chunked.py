import os
import re
import json
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken


def chunk_code_smart(python_code: str, model: str, max_tokens: int = 3000) -> list:
    """
    Zerlegt den Python-Code in (möglichst) sinnvolle Teilstücke (Chunks), 
    sodass keine Methoden oder Klassen geteilt werden, wenn es sich vermeiden lässt.

    Args:
        python_code (str): Vollständiger Python-Code als String.
        model (str): Name des zu verwendenden OpenAI-Modells, z.B. "gpt-3.5-turbo".
        max_tokens (int): Maximale Token-Anzahl pro Chunk.

    Returns:
        list: Liste von Strings, bei denen jeder String einen Chunk repräsentiert.
    """
    # Regex, um Funktions- oder Klassen-Definitionen zu erkennen:
    definition_regex = re.compile(r'^\s*(def|class|async\s+def)\s+')

    # Encoder für Tokens
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    lines = python_code.split('\n')

    chunked_code = []
    chunk_lines = []
    chunk_tokens = 0

    # Speichert den Index der letzten gefundenen Klassen-/Funktions-Definition
    # innerhalb des aktuellen Chunks
    last_def_boundary_idx = None  # relativer Index innerhalb des aktuellen Chunks

    for line_idx, line in enumerate(lines):
        # Token-Anzahl für diese Zeile
        line_token_count = len(encoding.encode(line + '\n'))

        # Prüfen, ob diese Zeile ein Funktions-/Klassen-Header ist
        if definition_regex.match(line):
            last_def_boundary_idx = len(chunk_lines)

        # Falls Hinzufügen dieser Zeile die Chunk-Grenze sprengt ...
        if chunk_tokens + line_token_count > max_tokens:
            # Versuch, an der letzten Funktions-/Klassendefinition zu trennen
            if last_def_boundary_idx is not None and last_def_boundary_idx > 0:
                # Wir können den Chunk bis zur letzten Definition sauber abschließen
                split_idx = last_def_boundary_idx
                # Hole den Teil bis zur letzten Definition (exklusiv) heraus
                chunked_code.append('\n'.join(chunk_lines[:split_idx]))

                # Den "Rest" bis zur aktuellen Zeile verschieben wir in den neuen Chunk
                # (die Definition selbst rutscht also in den nächsten Chunk)
                remaining_lines = chunk_lines[split_idx:]
                chunk_lines = remaining_lines[:]  # Kopie
                chunk_tokens = sum(len(encoding.encode(ln + '\n')) for ln in chunk_lines)
                # Jetzt fügen wir auch die aktuelle Zeile (die den Overflow verursacht hat) hinzu
                chunk_lines.append(line)
                chunk_tokens += line_token_count

            else:
                # Keine Definition gefunden oder Definition ist am Zeilenanfang 
                # -> Wir müssen direkt hier einen Cut machen
                if chunk_lines:
                    chunked_code.append('\n'.join(chunk_lines))
                # Neuer Chunk beginnt mit dieser Zeile
                chunk_lines = [line]
                chunk_tokens = line_token_count

            # Reset der Boundary
            last_def_boundary_idx = None
        else:
            # Alles okay, Zeile ganz normal anhängen
            chunk_lines.append(line)
            chunk_tokens += line_token_count

    # Letzten Chunk, falls vorhanden, anhängen
    if chunk_lines:
        chunked_code.append('\n'.join(chunk_lines))

    return chunked_code


def annotate_script(model: str, python_code: str) -> str:
    """
    Annotates a Python script with inline docstrings using the OpenAI API.
    """
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY", "keyhere")

    client = OpenAI(api_key=api_key)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant for analyzing Python code and generating inline docstrings.",
            },
            {
                "role": "user",
                "content": (
                    "Please add appropriate inline docstrings to the following Python code, "
                    "ensuring the file remains executable:\n\n"
                    f"{python_code}"
                ),
            },
        ],
        model=model,
    )

    annotated_code = chat_completion.choices[0].message.content

    # Entferne etwaige ```-Codeblöcke
    lines = annotated_code.split("\n")
    lines = [ln for ln in lines if not ln.strip().startswith("```")]
    annotated_code = "\n".join(lines)
    return annotated_code


def annotate_by_mapping(model: str, original_python_code: str, target_directory: str, file_name: str) -> str:
    """
    Umfasst:
    1) "Intelligentes" Chunking des Codes, um nicht mitten in einer Funktion/Klasse zu trennen.
    2) Erstellung von Docstrings nur für Klassen- und Funktionsheader pro Chunk.
    3) Zusammenführen der Annotationen.
    4) Entfernen vorhandener Docstrings.
    5) Einfügen der neu generierten Docstrings.
    6) Speichern des finalen Codes.
    """
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY", "keyhere")
    client = OpenAI(api_key=api_key)

    # 1) Code in Chunks zerlegen (intelligentes Chunking)
    chunked_code_list = chunk_code_smart(original_python_code, model=model, max_tokens=3000)

    all_generated_annotations = []

    # 2) Für jeden Chunk die Docstrings generieren
    for idx, chunk in enumerate(chunked_code_list, start=1):
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant for analyzing Python code and generating inline docstrings.",
                },
                {
                    "role": "user",
                    "content": (
                        "Please generate appropriate inline docstrings for both functions and classes "
                        "in the following Python code. Only return the class header, function header, "
                        "or asynchronous function header and the docstring. Retain indentation and basic structure. "
                        "Use triple quotes \"\"\" as a docstring wrapper.\n\n"
                        f"{chunk}"
                    ),
                },
            ],
            model=model,
        )

        chunk_generated = chat_completion.choices[0].message.content
        # Entferne ```-Blöcke
        lines = chunk_generated.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        chunk_generated_cleaned = "\n".join(lines)

        all_generated_annotations.append(chunk_generated_cleaned)

    # 3) Zusammenführen aller Teil-Annotationen
    combined_generated_annotations = "\n".join(all_generated_annotations)

    # Debug: speichere den kombinierten Text
    script_name_combined = file_name.replace(".py", "_generated_docstrings_combined.py")
    output_path_combined = os.path.join(target_directory, script_name_combined)
    with open(output_path_combined, "w", encoding="utf-8") as output_file:
        output_file.write(combined_generated_annotations)

    # 4) Entfernen vorhandener Docstrings (alte docstrings raus)
    cleaned_python_code = remove_docstrings(original_python_code)
    script_name_cleaned = file_name.replace(".py", "_cleaned.py")
    output_path_cleaned = os.path.join(target_directory, script_name_cleaned)
    with open(output_path_cleaned, "w", encoding="utf-8") as output_file:
        output_file.write(cleaned_python_code)

    # 5) Einfügen der neuen Docstrings
    annotated_code = map_annotations(cleaned_python_code, combined_generated_annotations, target_directory, file_name)

    # 6) Endergebnis speichern
    output_path = os.path.join(target_directory, file_name)
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(annotated_code)

    return annotated_code


def map_annotations(input_code: str, annotations: str, target_directory: str, file_name: str) -> str:
    """
    Verknüpft die generierten Docstrings mit den jeweiligen Klassen-/Funktions-Definitionen.
    """
    regex = r'(^\s*)(class|def|async def)(\s+)(\w+)(.*)'

    def parse_annotations(annotations_text: str, target_dir: str, f_name: str) -> dict:
        annotation_dict = {}
        current_key = None
        current_value = []
        for annotation_line in annotations_text.splitlines():
            annotation_line = annotation_line.rstrip()
            if re.match(regex, annotation_line):
                if current_key:
                    annotation_dict[current_key] = "\n".join(current_value).rstrip()
                current_key = re.match(regex, annotation_line).group(0)
                current_value = []
            elif current_key is not None:
                current_value.append(annotation_line)

        if current_key:
            annotation_dict[current_key] = "\n".join(current_value).rstrip()


        # Debug: Annotationen als JSON speichern
        script_name = f_name.replace(".py", "_JSON.json")
        output_path = os.path.join(target_dir, script_name)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(annotation_dict, f, indent=4)

        return annotation_dict

    annotation_dict = parse_annotations(annotations, target_directory, file_name)

    def insert_annotations(code_text: str, annotation_map: dict) -> str:
        code_lines = code_text.splitlines()
        annotated_code_lines = []

        for cline in code_lines:
            annotated_code_lines.append(cline)
            match = re.match(regex, cline)
            if match:
                header_line = match.group(0)
                if header_line in annotation_map:
                    annotated_code_lines.append(annotation_map[header_line])

        return "\n".join(annotated_code_lines)

    return insert_annotations(input_code, annotation_dict)


def remove_docstrings(code: str) -> str:
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


if __name__ == "__main__":
    model = "gpt-3.5-turbo"
#   with open("downloaded_files/quantumiracle/ppo_gae_continuous3.py", "r", encoding="utf-8") as file:
#   with open("downloaded_files/wesselb/readme_example8_gp-rnn.py", "r", encoding="utf-8") as file:
#   with open("downloaded_files/ChenRocks/training.py", "r", encoding="utf-8") as file:
#   with open("downloaded_files/streamlit/st_magic.py", "r", encoding="utf-8") as file:
    with open("downloaded_files/pyro-ppl/integrate.py", "r", encoding="utf-8") as file:
        python_code = file.read()

    annotate_by_mapping(model, python_code, "examples/run_C", "integrate.py")
