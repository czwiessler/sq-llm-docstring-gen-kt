from sample_generator import annotate_random_scripts



annotate_random_scripts(
    save_directory="examples/run_C",
    min_functions=5,
    max_functions=50,
    min_classes=2,
    max_classes=50,
    min_functions_with_docstring=0,
    max_functions_with_docstring=0,
    number_of_scripts=1,
    min_loc=50,
    max_loc=2500,
)