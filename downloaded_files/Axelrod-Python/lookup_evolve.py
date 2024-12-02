"""Lookup Table Evolver

Usage:
    lookup_evolve.py [-h] [--generations GENERATIONS] [--population POPULATION]
    [--mu MUTATION_RATE] [--bottleneck BOTTLENECK] [--processes PROCESSES]
    [--output OUTPUT_FILE] [--objective OBJECTIVE] [--repetitions REPETITIONS]
    [--turns TURNS] [--noise NOISE] [--nmoran NMORAN]
    [--plays PLAYS] [--op_plays OP_PLAYS] [--op_start_plays OP_START_PLAYS]

Options:
    -h --help                   Show help
    --generations GENERATIONS   Generations to run the EA [default: 500]
    --population POPULATION     Starting population size  [default: 40]
    --mu MUTATION_RATE          Mutation rate [default: 0.1]
    --bottleneck BOTTLENECK     Number of individuals to keep from each generation [default: 10]
    --processes PROCESSES       Number of processes to use [default: 1]
    --output OUTPUT_FILE        File to write data to [default: lookup_params.csv]
    --objective OBJECTIVE       Objective function [default: score]
    --repetitions REPETITIONS   Repetitions in objective [default: 100]
    --turns TURNS               Turns in each match [default: 200]
    --noise NOISE               Match noise [default: 0.00]
    --nmoran NMORAN             Moran Population Size, if Moran objective [default: 4]
    --plays PLAYS               Number of recent plays in the lookup table [default: 2]
    --op_plays OP_PLAYS         Number of recent plays in the lookup table [default: 2]
    --op_start_plays OP_START_PLAYS   Number of opponent starting plays in the lookup table [default: 2]
"""


from axelrod import Action, EvolvableLookerUp
from axelrod_dojo import invoke_training


def prepare_player_class_kwargs(arguments):
    plays = int(arguments['--plays'])
    op_plays = int(arguments['--op_plays'])
    op_start_plays = int(arguments['--op_start_plays'])
    table_depth = max(plays, op_plays, op_start_plays)
    initial_actions = [Action.C] * table_depth
    param_kwargs = {
        "parameters": (plays, op_plays, op_start_plays),
        "initial_actions": initial_actions,
        "mutation_probability": float(arguments['--mu']),
    }
    return param_kwargs


if __name__ == '__main__':
    invoke_training(
        __doc__,
        'LookerUp Evolver 0.4',
        EvolvableLookerUp,
        prepare_player_class_kwargs
    )