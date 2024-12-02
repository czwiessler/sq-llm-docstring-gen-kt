## Begleitmaterial für Projekt 8: "Automated Docstring Generation for Python Scripts"

### Beschreibung
Dieses Repository stellt optionales Begleitmaterial für das Projekt zur Verfügung. Es kann, muss aber nicht für das Projekt verwendet werden. Enthalten sind:
1. Ordner `downloaded_files`: 25563 Python Skripte aus öffentlich zugänglichen Github-Repositories zum Thema Machine Learning. Die jeweiligen Unterordner, in denen die Skripte liegen, bilden die Namen des entsprechenden Github-Repositories ab.
2. Skript `script_metrics.xlsx`: Diverse Metriken zu den Skripten (siehe unten).
3. Skript `filter_sample_metrics.py`: Python Tool zum Filtern der ML-Skripte (siehe unten).

### Erläuterung zu `script_metrics.xlsx`
- `file`: Dateiname
- `lines`: Anzahl Zeilen (inkl. Leerzeilen, Kommentare, etc.)
- `classes`: Anzahl der Klassen
- `functions`: Anzahl Funktionen
- `non_class_function_lines`: Anzahl der Zeilen die nicht zu einer Funktion gehören
- `nested_lines`: Anzahl der Zeilen, die eingerückt sind (also 1 oder mehr Leerzeichen zu Beginn der Zeile haben, einschließlich Codezeilen in Funktionen)
- `non_nested_lines`: Anzahl der Zeilen, die nicht eingerückt sind (also keine Leerzeichen zu Beginn der Zeile haben)
- `import_statements`: Anzahl Import-Statements
- `loops`: Summe der While- und For-Loops
- `if_statements`: Anzahl der If-Statements
- `variables`: Summe der zugewiesenen Variablen
- `files_read`: Summe der Stellen, an denen eine externe Date eingelesen wird (berücksichtigt Funktionsaufrufe mit dem Muster `.open()` und `.read()`
- `functions_with_docstring`: Funktionen, die einen Docstring enthalten
- `single_line_comments`: Summe der Zeilenkommentare
- `average_line_length`: Durchschnittliche Zeilenlänge
- `maximum_line_length`: Maximale Zeilenlänge
- `average_function_length`: Durchschnittliche Funktionslänge
- `max_function_length`: Maximale Funktionslänge

### Tool: `filter_sample_metrics.py`
Mit dem Tool können die oben genannten ML Skripte gefiltert werden. Das Tool liest die Datei `script_metrics.xlsx` ein und filtert diejenigen Zeilen, die den definierten Conditions entsprechen. Diese gefilterten Zeilen werden dann standardmäßig in `filtered_files.xlsx` aus gegeben.

Die Conditions werden als Lambda Function in einem Dictionary definiert. Als Key wird der Spaltenname in `script_metrics.xlsx` angegeben, als Value wird die Lambda Function eingetragen. Beispiele:

```python
conditions  = {
	"lines": lambda  x: 30  <=  x  <=  300, # between 30 and 300 lines
	"functions": lambda  x: x  >=  3, # at least three functions
	"functions_with_docstring": lambda  x: x  >=  3, # at least three functions with a docstring
	...
}
```