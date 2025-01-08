import os
from filter_sample_scripts import filter_files_based_on_conditions
from docstring_generator_functionized import annotate_script



def annotate_random_scripts(
        save_directory: str,
        min_functions: int,
        max_functions: int,
        min_classes: int,
        max_classes: int,
        min_functions_with_docstring: int,
        max_functions_with_docstring: int,
        model: str = "gpt-3.5-turbo",
        number_of_scripts: int = 5,
        min_loc: int = 30,
        max_loc: int = 200,

):
    """
    Annotates a random sample of Python scripts based on specified conditions.

    Args:
        save_directory (str): Directory to save the annotated scripts.
        min_functions (int): Minimum number of functions in the script.
        max_functions (int): Maximum number of functions in the script.
        min_classes (int): Minimum number of classes in the script.
        max_classes (int): Maximum number of classes in the script.
        min_functions_with_docstring (int): Minimum number of functions with a docstring in the script.
        max_functions_with_docstring (int): Maximum number of functions with a docstring in the script.
        model (str): OpenAI model to use for generating docstrings.
        number_of_scripts (int): Number of scripts to annotate.
        min_loc (int): Minimum number of lines of code in the script.
        max_loc (int): Maximum number of lines of code in the script.
    """

    # Define conditions for filtering scripts
    conditions = {
        "lines": lambda x: min_loc <= x <= max_loc,
        "functions": lambda x: min_functions <= x <= max_functions,
        "classes": lambda x: min_classes <= x <= max_classes,
        "functions_with_docstring": lambda x: min_functions_with_docstring <= x <= max_functions_with_docstring,
    }

    # Filter scripts based on conditions
    metrics_file = "./script_metrics.xlsx"
    output_file = "./filtered_files.xlsx"
    filtered_df = filter_files_based_on_conditions(metrics_file, conditions, output_file)

    # Check if enough scripts match the criteria
    if len(filtered_df) < number_of_scripts:
        raise ValueError("Not enough scripts match the specified conditions.")

    # Randomly sample the specified number of scripts
    sampled_files = filtered_df.sample(n=number_of_scripts)["file"].tolist()

    # Annotate each sampled script and save the result
    for script_path in sampled_files:
        if not os.path.isfile(script_path):
            print(f"Warning: File {script_path} not found. Skipping.")
            continue

        # Annotate the script
        with open(script_path, "r", encoding="utf-8") as file:
            script_content = file.read()

        annotated_content = annotate_script(model, script_content)

        # Save the annotated script
        script_name = os.path.basename(script_path)
        output_path = os.path.join(save_directory, script_name)
        with open(output_path, "w", encoding="utf-8") as output_file:
            output_file.write(annotated_content)

        print(f"Annotated script saved to {output_path}")
