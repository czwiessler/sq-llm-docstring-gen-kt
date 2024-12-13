"""
ANN evolver.
Trains ANN strategies with an evolutionary algorithm.

Original version by Martin Jones @mojones:
https://gist.github.com/mojones/b809ba565c93feb8d44becc7b93e37c6

Usage:
    ann_evolve.py [-h] [--generations GENERATIONS] [--population POPULATION]
    [--mu mutation_probability] [--bottleneck BOTTLENECK] [--processes PROCESSES]
    [--output OUTPUT_FILE] [--objective OBJECTIVE] [--repetitions REPETITIONS]
    [--turns TURNS] [--noise NOISE] [--nmoran NMORAN]
    [--features FEATURES] [--hidden HIDDEN] [--mu_distance DISTANCE]

Options:
    -h --help                   Show help
    --generations GENERATIONS   Generations to run the EA [default: 500]
    --population POPULATION     Starting population size  [default: 40]
    --mu mutation_probability          Mutation rate [default: 0.1]
    --bottleneck BOTTLENECK     Number of individuals to keep from each generation [default: 10]
    --processes PROCESSES       Number of processes to use [default: 1]
    --output OUTPUT_FILE        File to write data to [default: ann_params.csv]
    --objective OBJECTIVE       Objective function [default: score]
    --repetitions REPETITIONS   Repetitions in objective [default: 100]
    --turns TURNS               Turns in each match [default: 200]
    --noise NOISE               Match noise [default: 0.00]
    --nmoran NMORAN             Moran Population Size, if Moran objective [default: 4]
    --features FEATURES         Number of ANN features [default: 17]
    --hidden HIDDEN             Number of hidden nodes [default: 10]
    --mu_distance DISTANCE      Delta max for weights updates [default: 10]
"""

from axelrod import EvolvableANN
from axelrod_dojo import invoke_training


def prepare_player_class_kwargs(arguments):
    param_kwargs = {
        "num_features": int(arguments['--features']),
        "num_hidden": int(arguments['--hidden']),
        "mutation_probability": float(arguments['--mu']),
        "mutation_distance": float(arguments['--mu_distance'])
    }
    return param_kwargs


if __name__ == '__main__':
    invoke_training(
        __doc__,
        'ANN Evolver 0.4',
        EvolvableANN,
        prepare_player_class_kwargs
    )
