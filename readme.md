# README: **DockstringGenSimple**

## Overview

**DockstringGenSimple** is a lightweight tool designed to automatically generate docstrings for Python files. 
## Installation

To install the necessary dependencies, ensure you have Python installed and then run:

```bash
pip install -r requirements.txt
```

## Usage

To generate docstrings for a given Python file, simply execute the following command:

```bash
py docstring_gen_single.py <path_to_python_file>
```

Upon execution, the tool processes the specified Python script and creates an annotated version in the same directory. The new file is saved under the same name with the suffix `_annotated.py`. 

### Example

If you have a script named `example.py`, running:

```bash
python docstring_gen_single.py example.py
```

will generate an annotated version:

```
example_annotated.py
```

## Dependencies

All required dependencies are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

## License

This project is released under the [MIT License](LICENSE). 

