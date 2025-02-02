import os
import re
import json
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken
from typing import List, Dict

def remove_code_fences(text: str) -> str:
    """
    Entfernt dreifache Backticks ```...``` aus einem Text,
    die ggf. vom Modell um generierten Code verwendet werden.
    """
    lines = text.split("\n")
    cleaned = [ln for ln in lines if not ln.strip().startswith("```")]
    return "\n".join(cleaned)


def chunk_code_smart(
    python_code: str,
    model: str,
    max_tokens: int = 3000
) -> List[str]:
    """
    Zerlegt den Python-Code in (möglichst) sinnvolle Teilstücke (Chunks),
    sodass keine Methoden oder Klassen geteilt werden, wenn es sich vermeiden lässt.
    """
    definition_regex = re.compile(r'^\s*(def|class|async\s+def)\s+')
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    lines = python_code.split('\n')

    chunked_code = []
    chunk_lines = []
    chunk_tokens = 0

    last_def_boundary_idx = None

    for line_idx, line in enumerate(lines):
        line_token_count = len(encoding.encode(line + '\n'))

        if definition_regex.match(line):
            last_def_boundary_idx = len(chunk_lines)

        if chunk_tokens + line_token_count > max_tokens:
            if last_def_boundary_idx is not None and last_def_boundary_idx > 0:
                # Split at the last class/def boundary
                split_idx = last_def_boundary_idx
                chunked_code.append('\n'.join(chunk_lines[:split_idx]))

                remaining_lines = chunk_lines[split_idx:]
                chunk_lines = remaining_lines[:]
                chunk_tokens = sum(len(encoding.encode(ln + '\n')) for ln in chunk_lines)

                chunk_lines.append(line)
                chunk_tokens += line_token_count
            else:
                if chunk_lines:
                    chunked_code.append('\n'.join(chunk_lines))

                chunk_lines = [line]
                chunk_tokens = line_token_count

            last_def_boundary_idx = None
        else:
            chunk_lines.append(line)
            chunk_tokens += line_token_count

    if chunk_lines:
        chunked_code.append('\n'.join(chunk_lines))

    return chunked_code


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


def parse_annotations(
    annotations_text: str,
    target_dir: str,
    f_name: str,
    debug: bool = False
) -> Dict[str, str]:
    """
    Zerlegt den generierten Annotationstext (Docstrings) in ein Dictionary:
      - Key: exakte Header-Zeile (z.B. "def foo(...):")
      - Value: der zugehörige Docstring
    """
    header_regex = re.compile(r'^\s*(?:class|def|async\s+def)\s+')
    decorator_regex = re.compile(r'^\s*@')

    annotation_dict = {}
    current_key = None
    current_value = []

    lines = annotations_text.splitlines()
    for idx, line in enumerate(lines):
        line_stripped = line.rstrip()

        # 1) Neuer Funktions-/Klassen-Header?
        if header_regex.match(line_stripped):
            if current_key is not None:
                annotation_dict[current_key] = "\n".join(current_value).rstrip()
                #print(f"[INFO] Block abgeschlossen: {current_key}")

            current_key = line_stripped
            current_value = []
            #print(f"[INFO] Neuer Header erkannt in Zeile {idx + 1}: {current_key}")

        # 2) Dekorator-Zeile
        elif decorator_regex.match(line_stripped):
            if current_key is not None:
                annotation_dict[current_key] = "\n".join(current_value).rstrip()
                #print(f"[INFO] Dekorator erkannt → Block abgeschlossen: {current_key}")
                current_key = None
                current_value = []
            #print(f"[INFO] Dekorator erkannt in Zeile {idx + 1}: {line_stripped}")

        else:
            # Normale Zeile (Teil des Docstrings)
            if current_key is not None:
                current_value.append(line_stripped)

    # Letzten Block sichern
    if current_key and current_value:
        annotation_dict[current_key] = "\n".join(current_value).rstrip()
        #print(f"[INFO] Letzter Block abgeschlossen: {current_key}")
    cleaned_data = {}
    for key, value in annotation_dict.items():
        # Behalte nur den Text zwischen den ersten und letzten """
        match = re.search(r'(\s*""".*?""")', value, flags=re.DOTALL)
        if match:
            cleaned_data[key] = match.group(1)
        else:
            cleaned_data[key] = value  # Falls kein Match, bleibt der Wert unverändert
    # Debug-Ausgabe
    if debug:
        script_name = f_name.replace(".py", "_JSON.json")
        output_path = os.path.join(target_dir, script_name)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_data, f, indent=4)
        #print(f"[DEBUG] JSON-Datei geschrieben: {output_path}")

    #print(f"[INFO] Insgesamt {len(annotation_dict)} Funktionen/Klassen erkannt.")
    return cleaned_data


def insert_annotations(
    code_text: str,
    annotation_map: Dict[str, str],
    regex_pattern: str
) -> str:
    """
    Geht zeilenweise durch den (bereinigten) Code und fügt
    den passenden Docstring direkt nach dem Header ein.

    Args:
        code_text (str): Code ohne alte Docstrings.
        annotation_map (dict): Dict mit {header_line: docstring_text}.
        regex_pattern (str): Regex zum Erkennen der Header-Zeile.

    Returns:
        str: Code mit neu eingefügten Docstrings.
    """
    code_lines = code_text.splitlines()
    annotated_code_lines = []

    header_regex = re.compile(regex_pattern)

    for line in code_lines:
        annotated_code_lines.append(line)
        match = header_regex.match(line)
        if match:
            # exakte Zeile inkl. Leerzeichen
            header_line = line.rstrip()
            if header_line in annotation_map:
                # Docstring einfügen
                annotated_code_lines.append(annotation_map[header_line])
        # falls alles übereistimmt, aber teilweise die leerzeichen nicht übereinstimmen, dann auch einfügen
        elif line.strip() in annotation_map:
            # TODO: das funzt noch nicht
            annotated_code_lines.append(annotation_map[line.strip()])
        # falls die headerzeile in code_text sich über mehrere zeilen erstreckt, in der annotation_map aber in einer zeile steht, dann nach dem : im zu anotierenden code eine neue zeile einfügen und dann den docstring
        # TODO: das funzt noch nicht
        elif line.strip().split(":")[0] in annotation_map:
            annotated_code_lines.append("\n" + annotation_map[line.strip().split(":")[0]])



    return "\n".join(annotated_code_lines)


# -------------------------------------------------------------------------
# NEW: Extract all definitions (class|def|async def) from code so we can “cache” them
# -------------------------------------------------------------------------
def extract_definitions_from_code(code: str) -> List[str]:
    """
    Sucht im Code nach allen Klassendefinitionen (class) und Funktions-/Methodendefinitionen (def, async def).
    Gibt die exakten Kopfzeilen zurück (inkl. Einrückung), sodass wir sie später 1:1 matchen können.
    """
    header_regex = re.compile(r'^(\s*)(class|def|async\s+def)\s+')
    lines = code.splitlines()

    definitions = []
    for line in lines:
        # Genau das, was wir später als Key verwenden wollen
        if header_regex.match(line):
            # Speichere die komplette Zeile (inkl. trailing spaces)
            # or at least rstrip() so we don't have issues with trailing spaces.
            # But it's crucial that the indentation + the actual definition is intact.
            definitions.append(line.rstrip())

    return definitions


# -------------------------------------------------------------------------
# NEW: A function that takes a chunk of code and asks the LLM for docstrings
# -------------------------------------------------------------------------
def generate_docstrings_for_chunk(
    client: OpenAI,
    model: str,
    chunk_code: str,
    system_prompt: str,
    user_prompt: str
) -> str:
    """
    Ruft die OpenAI-API auf, um für einen Code-Chunk die entsprechenden Docstrings zu generieren.
    Erzeugt einen Text, der nur (header + docstring) Zeilen enthält.
    """
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt + chunk_code},
        ],
        model=model,
    )

    chunk_generated = response.choices[0].message.content
    chunk_generated_cleaned = remove_code_fences(chunk_generated)
    return chunk_generated_cleaned


def annotate_by_mapping(
    model: str,
    original_python_code: str,
    target_directory: str,
    file_name: str,
    max_chunk_tokens: int = 3000,
    debug: bool = True
) -> str:
    """
    1) Chunking des Codes (falls nötig).
    2) Extrahieren aller class-/def-/async def-Headers → "Cache".
    3) Für jeden Chunk Docstrings generieren (via LLM).
    4) Zusammenführen dieser generierten Docstring-Schnipsel.
    5) Alte Docstrings entfernen.
    6) Neue Docstrings einfügen (map_annotations).
    7) Speichern des finalen Codes (plus Debugfiles, wenn debug=True).
    """
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY", "keyhere")
    client = OpenAI(api_key=api_key)

    # ---------------------------------------------------------
    # 1) Code in Chunks zerlegen
    # ---------------------------------------------------------
    chunked_code_list = chunk_code_smart(
        python_code=original_python_code,
        model=model,
        max_tokens=max_chunk_tokens
    )
    print(f"[INFO] Code in {len(chunked_code_list)} Chunks zerlegt.")

    # ---------------------------------------------------------
    # 2) Alle Definitionen extrahieren (globaler Überblick)
    # ---------------------------------------------------------
    all_definitions = extract_definitions_from_code(original_python_code)

    if debug:
        # Schreib sie z.B. in eine JSON-/Textdatei zum "Nachschauen"
        definitions_path = os.path.join(target_directory, file_name.replace(".py", "_definitions.txt"))
        with open(definitions_path, "w", encoding="utf-8") as f:
            for d in all_definitions:
                f.write(d + "\n")

    # ---------------------------------------------------------
    # 3) Docstrings generieren, chunk-weise
    # ---------------------------------------------------------
    system_prompt = (
        "You are a helpful assistant for analyzing Python code and generating inline docstrings."
    )
    user_prompt = (
        "Please generate appropriate inline docstrings for the folowing methods and classes "
        f"{all_definitions} "
        "in the following Python code. Only return the class header, function header, "
        "or asynchronous function header and the docstring. Retain indentation and basic structure. "
        "Use triple quotes \"\"\" as a docstring wrapper. This is very important.\n\n"
    )

    all_generated_annotations = []
    for idx, chunk in enumerate(chunked_code_list, start=1):
        # Hole Docstrings für den aktuellen Chunk
        chunk_generated_docstrings = generate_docstrings_for_chunk(
            client=client,
            model=model,
            chunk_code=chunk,
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        # Sammle sie
        all_generated_annotations.append(chunk_generated_docstrings)

    # 3b) Alles zusammenführen
    combined_generated_annotations = "\n".join(all_generated_annotations)

    # Debug: Speichern der kombinierten Annotationen
    if debug:
        script_name_combined = file_name.replace(".py", "_generated_docstrings_combined.py")
        output_path_combined = os.path.join(target_directory, script_name_combined)
        with open(output_path_combined, "w", encoding="utf-8") as output_file:
            output_file.write(combined_generated_annotations)

    # ---------------------------------------------------------
    # 4) Alte Docstrings entfernen
    # ---------------------------------------------------------
    cleaned_python_code = remove_docstrings(original_python_code)
    if debug:
        script_name_cleaned = file_name.replace(".py", "_cleaned.py")
        output_path_cleaned = os.path.join(target_directory, script_name_cleaned)
        with open(output_path_cleaned, "w", encoding="utf-8") as output_file:
            output_file.write(cleaned_python_code)

    # ---------------------------------------------------------
    # 5) Neue Docstrings einfügen
    # ---------------------------------------------------------
    # Zuerst parse_annotations → Dictionary {header_line: docstring}
    annotation_dict = parse_annotations(
        annotations_text=combined_generated_annotations,
        target_dir=target_directory,
        f_name=file_name,
        debug=debug
    )

    # Dann per insert_annotations wieder einfügen
    # Regex für "class X:" / "def X(...):" / "async def X(...):"
    regex = r'(^\s*)(class|def|async def)(\s+)(\w+)(.*)'
    annotated_code = insert_annotations(
        code_text=cleaned_python_code,
        annotation_map=annotation_dict,
        regex_pattern=regex
    )

    # ---------------------------------------------------------
    # 6) Endergebnis speichern
    # ---------------------------------------------------------
    output_path = os.path.join(target_directory, file_name)
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(annotated_code)

    return annotated_code


if __name__ == "__main__":
    model_name = "gpt-3.5-turbo"

    # Beispiel: Datei einlesen
    with open(r"downloaded_files\tbepler\extract.py", "r", encoding="utf-8") as file:
        python_code_example = file.read()

    # Annotieren
    annotate_by_mapping(
        model=model_name,
        original_python_code=python_code_example,
        target_directory="examples/run_C",
        file_name="extract.py",
        max_chunk_tokens=50000,
        debug=True
    )
