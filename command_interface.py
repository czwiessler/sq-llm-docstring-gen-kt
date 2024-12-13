from sample_generator import annotate_random_scripts



annotate_random_scripts(
    save_directory="examples/run_A",
    min_functions=2,
    max_functions=2,
    min_functions_with_docstring=0,
    max_functions_with_docstring=5,
    number_of_scripts=1,
    min_loc=50,
    max_loc=50,
)