import os
from openai import OpenAI

# OpenAI API-Schlüssel aus Umgebungsvariablen lesen oder direkt eingeben
api_key = os.environ.get("OPENAI_API_KEY", "keyhere")

# OpenAI-Client initialisieren
client = OpenAI(api_key=api_key)

# Datei lesen
file_path = "example.py"

with open(file_path, "r", encoding="utf-8") as file:
    python_code = file.read()

# Anfrage an die Chat-Completion API
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
    model="gpt-3.5-turbo",
)

# Generierten Inhalt extrahieren
generated_docstrings = chat_completion.choices[0].message.content

# Speichern des generierten Codes in einer neuen Datei
output_path = "example_with_docstrings.py"
with open(output_path, "w", encoding="utf-8") as output_file:
    output_file.write(generated_docstrings)

print(f"Docstrings wurden hinzugefügt und unter {output_path} gespeichert.")