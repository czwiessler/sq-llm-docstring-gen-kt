import os
import re
import json
from dotenv import load_dotenv
from openai import OpenAI

# Neu importiert:
import tiktoken


def chunk_code(python_code: str, model: str, max_tokens: int = 3000) -> list:
    """
    Zerlegt den Python-Code in kleinere Teilstücke (Chunks), basierend auf einer
    Token-Grenze, die wir mit Hilfe von tiktoken abschätzen.
    Zeilenbasierter Ansatz: Bei Überschreiten von max_tokens wird ein neuer Chunk begonnen.

    Args:
        python_code (str): Vollständiger Python-Code als String.
        model (str): Name des zu verwendenden OpenAI-Modells, z.B. "gpt-3.5-turbo".
        max_tokens (int): Maximale Token-Anzahl pro Chunk.

    Returns:
        list: Liste von Strings, bei denen jeder String einen Chunk des ursprünglichen Codes repräsentiert.
    """
    # tiktoken-Encoder für das ausgewählte Modell laden
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback, falls das Modell nicht direkt erkannt wird
        encoding = tiktoken.get_encoding("cl100k_base")

    lines = python_code.split('\n')
    chunked_code = []
    current_chunk = []
    current_chunk_tokens = 0

    for line in lines:
        # +1 wegen Zeilenumbruch
        line_tokens = len(encoding.encode(line + '\n'))

        # Wenn das Hinzufügen der aktuellen Zeile den max_tokens-Rahmen sprengen würde:
        if current_chunk_tokens + line_tokens > max_tokens:
            # alten Chunk speichern
            chunked_code.append('\n'.join(current_chunk))
            # neuer Chunk beginnt
            current_chunk = [line]
            current_chunk_tokens = line_tokens
        else:
            # Zeile zum aktuellen Chunk hinzufügen
            current_chunk.append(line)
            current_chunk_tokens += line_tokens

    # Der letzte Chunk, falls vorhanden
    if current_chunk:
        chunked_code.append('\n'.join(current_chunk))

    return chunked_code


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

    # Entferne ggf. Codeblöcke ```...```
    annotated_code = annotated_code.split("\n")
    annotated_code = [line for line in annotated_code if not line.strip().startswith("```")]
    annotated_code = "\n".join(annotated_code)

    return annotated_code


def annotate_by_mapping(model: str, original_python_code: str, target_directory: str, file_name: str) -> str:
    """
    Annotiert einen Python-Code in mehreren Schritten:
      1. Zerlegung (Chunking) des Codes (falls zu groß) mittels tiktoken
      2. Für jeden Chunk: Generierung von Docstrings nur für Funktions- und Klassenheader
      3. Zusammenfügen der einzelnen Docstring-Ergebnisse
      4. Entfernen vorhandener Docstrings aus dem Originalcode
      5. Einfügen der generierten Docstrings in den bereinigten Code
      6. Speichern des final annotierten Codes

    Args:
        model (str): Das OpenAI-Modell, z.B. "gpt-3.5-turbo"
        original_python_code (str): Der zu annotierende Python-Code
        target_directory (str): Zielverzeichnis zum Speichern von Debug-/Output-Files
        file_name (str): Dateiname der zu annotierenden Python-Datei

    Returns:
        str: Der final annotierte Python-Code
    """
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY", "keyhere")
    client = OpenAI(api_key=api_key)

    # 1) Code in Chunks zerlegen, damit wir bei großen Dateien nicht die Token-Grenze sprengen
    chunked_code_list = chunk_code(original_python_code, model=model, max_tokens=3000)

    all_generated_annotations = []

    # 2) Für jeden Chunk Docstrings generieren (nur für Funktions- und Klassenheader)
    for idx, chunk in enumerate(chunked_code_list, start=1):
        # Prompt wie zuvor, aber nur für diesen Chunk
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant for analyzing Python code and generating inline docstrings.",
                },
                {
                    "role": "user",
                    "content": (
                        "Please generate appropriate inline docstrings for both functions and classes in the following Python code. "
                        "Only return the class header, function header, or asynchronous function header and the docstring; "
                        "retain indentation and basic structure. Use triple quotes \"\"\" as a docstring wrapper.\n\n"
                        f"{chunk}"
                    ),
                },
            ],
            model=model,
        )

        chunk_generated = chat_completion.choices[0].message.content
        # Entferne ggf. ```...```-Codeblöcke
        lines = chunk_generated.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        chunk_generated_cleaned = "\n".join(lines)

        # Sammeln aller generierten Annotationen
        all_generated_annotations.append(chunk_generated_cleaned)

    # 3) Kombiniere alle Docstring-Resultate der Chunks zu einem großen String
    combined_generated_annotations = "\n".join(all_generated_annotations)

    # Debug: Schreibe den kombinierten Ergebnis-String in eine Datei (optional)
    script_name_combined = file_name.replace(".py", "_generated_docstrings_combined.py")
    output_path_combined = os.path.join(target_directory, script_name_combined)
    with open(output_path_combined, "w", encoding="utf-8") as output_file:
        output_file.write(combined_generated_annotations)

    # 4) Entferne evtl. bereits vorhandene Docstrings aus dem Original
    cleaned_python_code = remove_docstrings(original_python_code)

    # Debug: Schreibe den bereinigten Code
    script_name_cleaned = file_name.replace(".py", "_cleaned.py")
    output_path_cleaned = os.path.join(target_directory, script_name_cleaned)
    with open(output_path_cleaned, "w", encoding="utf-8") as output_file:
        output_file.write(cleaned_python_code)

    # 5) Füge die neuen Docstrings in den Code ein
    annotated_code = map_annotations(cleaned_python_code, combined_generated_annotations, target_directory, file_name)

    # 6) Final speichern
    output_path = os.path.join(target_directory, file_name)
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(annotated_code)

    return annotated_code


def map_annotations(input_code: str, annotations: str, target_directory: str, file_name: str) -> str:
    """
    Sucht anhand einer Regex die Klassen- und Funktionsheader im Code und mapped
    die generierten Docstrings (in `annotations`) an die entsprechende Stelle.
    """
    regex = r'(^\s*)(class|def|async def)(\s+)(\w+)(.*)'

    def parse_annotations(annotations_text: str, target_directory: str, file_name: str) -> dict:
        """
        Zerlegt den gesamten (kombinierten) Annotation-Text in einen Dictionary-Mapping:
          - Key: die Zeile mit dem Header (z.B. `def my_function(...):`)
          - Value: der dazugehörige Docstring (ggf. mehrzeilig)
        """
        annotation_dict = {}
        current_key = None
        current_value = []

        for annotation_line in annotations_text.splitlines():
            annotation_line = annotation_line.rstrip()
            # neuer Header?
            if re.match(regex, annotation_line):
                # vorherigen Key abspeichern
                if current_key:
                    annotation_dict[current_key] = "\n".join(current_value).rstrip()
                current_key = re.match(regex, annotation_line).group(0)
                current_value = []
            elif current_key is not None:
                current_value.append(annotation_line)

        if current_key:
            annotation_dict[current_key] = "\n".join(current_value).rstrip()

        # Debug: Das Annotation-Dict nochmal als JSON speichern
        script_name = file_name.replace(".py", "_JSON.json")
        output_path = os.path.join(target_directory, script_name)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(annotation_dict, f, indent=4)

        return annotation_dict

    annotation_dict = parse_annotations(annotations, target_directory, file_name)

    def insert_annotations(input_text: str, annotation_dict_: dict) -> str:
        """
        Geht zeilenweise durch den bereinigten Code und fügt
        an der richtigen Stelle den entsprechenden Docstring-Block an.
        """
        code_lines = input_text.splitlines()
        annotated_code_lines = []

        for code_line in code_lines:
            annotated_code_lines.append(code_line)
            match = re.match(regex, code_line)
            if match:
                whole_first_header_line = match.group(0)
                if whole_first_header_line in annotation_dict_:
                    annotated_code_lines.append(annotation_dict_[whole_first_header_line])

        return "\n".join(annotated_code_lines)

    return insert_annotations(input_code, annotation_dict)


def remove_docstrings(code: str) -> str:
    """
    Entfernt zunächst alle Docstrings und fügt dann (optional) den
    ursprünglichen Modul-Docstring wieder ein, falls vorhanden.
    """
    # Regulärer Ausdruck für den Modul-Docstring
    module_docstring_pattern = r'\A\s*("""|\'\'\')((?:.|\n)*?)\1'
    module_match = re.match(module_docstring_pattern, code, flags=re.DOTALL)
    module_docstring = module_match.group(0) if module_match else ''

    # Regulärer Ausdruck für alle Docstrings
    all_docstring_pattern = r'("""|\'\'\')((?:.|\n)*?)\1'

    # Entfernt alle Docstrings
    cleaned_code = re.sub(all_docstring_pattern, '', code, flags=re.DOTALL)

    # Falls ein Modul-Docstring existiert, wieder vorne anhängen
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
