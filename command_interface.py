from sample_generator import annotate_random_scripts



annotate_random_scripts(
    save_directory="examples/run_C",
    min_functions=1,
    max_functions=5,
    min_classes=1,
    max_classes=4,
    min_functions_with_docstring=0,
    max_functions_with_docstring=0,
    number_of_scripts=8,
    min_loc=50,
    max_loc=250,
)