from sample_generator import annotate_random_scripts



annotate_random_scripts(
    save_directory="examples/run_A",
    min_functions=2,
    max_functions=99999,
    min_functions_with_docstring=0,
    max_functions_with_docstring=9999999,
    number_of_scripts=3,
    min_loc=100,
    max_loc=999999999,
)