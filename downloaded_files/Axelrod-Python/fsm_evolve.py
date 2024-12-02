"""
Finite State Machine Evolver

Usage:
    fsm_evolve.py [-h] [--generations GENERATIONS] [--population POPULATION]
    [--mu MUTATION_RATE] [--bottleneck BOTTLENECK] [--processes PROCESSES]
    [--output OUTPUT_FILE] [--objective OBJECTIVE] [--repetitions REPETITIONS]
    [--turns TURNS] [--noise NOISE] [--nmoran NMORAN]
    [--states NUM_STATES]

Options:
    -h --help                   Show help
    --generations GENERATIONS   Generations to run the EA [default: 500]
    --population POPULATION     Population size  [default: 40]
    --mu MUTATION_RATE          Mutation rate [default: 0.1]
    --bottleneck BOTTLENECK     Number of individuals to keep from each generation [default: 10]
    --processes PROCESSES       Number of processes to use [default: 1]
    --output OUTPUT_FILE        File to write data to [default: fsm_params.csv]
    --objective OBJECTIVE       Objective function [default: score]
    --repetitions REPETITIONS   Repetitions in objective [default: 100]
    --turns TURNS               Turns in each match [default: 200]
    --noise NOISE               Match noise [default: 0.00]
    --nmoran NMORAN             Moran Population Size, if Moran objective [default: 4]
    --states NUM_STATES         Number of FSM states [default: 8]
"""

from axelrod import EvolvableFSMPlayer
from axelrod_dojo import invoke_training


def prepare_player_class_kwargs(arguments):
    param_kwargs = {
        "num_states": int(arguments['--states']),
        "mutation_probability": float(arguments['--mu']),
    }
    return param_kwargs


if __name__ == '__main__':
    invoke_training(
        __doc__,
        'FSM Evolver 0.4',
        EvolvableFSMPlayer,
        prepare_player_class_kwargs
    )
