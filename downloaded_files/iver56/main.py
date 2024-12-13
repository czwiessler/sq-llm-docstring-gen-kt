import argparse
import neuroevolution
import experiment
import analyze
import settings


def positive_int(value):
    int_value = int(value)
    if int_value <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return int_value


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '-i',
        '--input',
        dest='input_files',
        nargs='+',
        type=str,
        help='The filename of the target sound and'
             ' the filename of the input sound, respectively',
        required=True,
        default=[]
    )
    arg_parser.add_argument(
        '-g',
        '--num-generations',
        dest='num_generations',
        type=positive_int,
        required=False,
        default=20
    )
    arg_parser.add_argument(
        '-p',
        '--population_size',
        dest='population_size',
        type=positive_int,
        required=False,
        default=20
    )
    arg_parser.add_argument(
        '--patience',
        dest='patience',
        help='Number of generations with no improvement in similarity before stopping',
        type=positive_int,
        required=False,
        default=100
    )
    arg_parser.add_argument(
        '-s',
        '--seed',
        dest='seed',
        help='PRNG seed. Will be set to a random value if not specified.',
        type=int,
        required=False,
        default=-1  # -1 means the seed will be random for each run
    )
    arg_parser.add_argument(
        '--keep-k-best',
        dest='keep_k_best',
        help='Store only the k fittest individual in each generation. Improves perf and'
             ' saves storage. If set to 0, no individuals will be stored.',
        type=int,
        required=False,
        default=-1  # -1 means keep all
    )
    arg_parser.add_argument(
        '--keep-all-last',
        nargs='?',
        dest='keep_all_last',
        help='Store all individuals in the last generation, disregarding --keep-k-best in'
             ' that generation',
        const=True,
        required=False,
        default=False
    )
    arg_parser.add_argument(
        '--keep-csd',
        nargs='?',
        dest='keep_csd',
        help='Keep all csd files that were used to generate the individuals. This will allow you to'
             ' run them with other input sounds',
        const=True,
        required=False,
        default=False
    )
    arg_parser.add_argument(
        '--verbose',
        nargs='?',
        dest='verbose',
        help='Activate more verbose output. Useful for debugging.',
        const=True,
        required=False,
        default=False
    )
    arg_parser.add_argument(
        '--allow-clones',
        nargs='?',
        dest='allow_clones',
        help="""Allow clones or nearly identical genomes to exist simultaneously in the
                    population. This is useful for non-deterministic environments,
                    as the same individual will get more than one chance to prove himself, also
                    there will be more chances the same individual to mutate in different ways.
                    The drawback is greatly increased time for reproduction. If you want to
                    search quickly, yet less efficient, leave this to true.""",
        const=True,
        required=False,
        default=False
    )
    arg_parser.add_argument(
        '--crossover-rate',
        dest='crossover_rate',
        type=float,
        help='CrossoverRate: Probability for a baby to result from sexual reproduction'
             ' (crossover/mating). 1.0 = 100 %%',
        required=False,
        default=0.7
    )
    arg_parser.add_argument(
        '--mutation-rate',
        dest='mutation_rate',
        type=float,
        help='OverallMutationRate: If a baby results from sexual reproduction, this probability'
             ' determines if mutation will be performed after crossover.'
             ' 1.0 = 100 %% (always mutate after crossover)',
        required=False,
        default=0.55
    )
    arg_parser.add_argument(
        '--add-neuron-prob',
        dest='add_neuron_probability',
        type=float,
        help='MutateAddNeuronProb: Probability for a baby to be mutated with the'
             ' Add-Neuron mutation',
        required=False,
        default=0.01
    )
    arg_parser.add_argument(
        '--add-link-prob',
        dest='add_link_probability',
        type=float,
        help='MutateAddLinkProb: Probability for a baby to be mutated with the'
             ' Add-Link mutation',
        required=False,
        default=0.01
    )
    arg_parser.add_argument(
        '--rem-link-prob',
        dest='remove_link_probability',
        type=float,
        help='MutateRemLinkProb: Probability for a baby to be mutated with the'
             ' Remove-Link mutation',
        required=False,
        default=0.01
    )
    arg_parser.add_argument(
        '--rem-simple-neuron-prob',
        dest='remove_simple_neuron_probability',
        type=float,
        help='MutateRemSimpleNeuronProb: Probability for a baby that a simple neuron'
             ' will be replaced with a link',
        required=False,
        default=0.01
    )
    arg_parser.add_argument(
        '--max-weight',
        dest='max_weight',
        type=float,
        help='MaxWeight: Maximum absolute magnitude of a weight',
        required=False,
        default=8.0
    )
    arg_parser.add_argument(
        '--weight-mutation-max-power',
        dest='weight_mutation_max_power',
        type=float,
        help='WeightMutationMaxPower: Maximum perturbation for a weight mutation',
        required=False,
        default=1.0
    )
    arg_parser.add_argument(
        '--mutate-weights-severe-prob',
        dest='mutate_weights_severe_prob',
        type=float,
        help='MutateWeightsSevereProb: Probability for a severe (shaking) weight mutation',
        required=False,
        default=0.25
    )
    arg_parser.add_argument(
        '--weight-mutation-rate',
        dest='weight_mutation_rate',
        type=float,
        help='WeightMutationRate: Probability for a particular gene\'s weight to be mutated.',
        required=False,
        default=1.0
    )
    arg_parser.add_argument(
        '--interspecies-crossover-rate',
        dest='interspecies_crossover_rate',
        type=float,
        help='InterspeciesCrossoverRate: Probability for a baby to result from'
             ' inter-species mating',
        required=False,
        default=0.0001
    )
    arg_parser.add_argument(
        '--elitism',
        dest='elitism',
        type=float,
        help='Fraction of population to carry on to the next generation unaltered',
        required=False,
        default=0.1
    )
    arg_parser.add_argument(
        '--survival-rate',
        dest='survival_rate',
        type=float,
        help='Fraction of best individuals that are allowed to reproduce',
        required=False,
        default=0.25
    )
    arg_parser.add_argument(
        '--fs-neat',
        nargs='?',
        dest='fs_neat',
        help='Use FS-NEAT (automatic feature selection)',
        const=True,
        required=False,
        default=False
    )
    arg_parser.add_argument(
        '--neural-mode',
        dest='neural_mode',
        type=str,
        choices=['a', 'ab', 'b', 's', 'targets'],
        help='Mode a: target sound is neural input.'
             ' Mode ab: target sound and input sound is neural input.'
             ' Mode b: input sound is neural input.'
             ' Mode s: static input, i.e. only bias.'
             ' Mode targets: evolve targets separately for each time step, with only static input',
        required=False,
        default="a"
    )
    arg_parser.add_argument(
        '--output-activation-function',
        dest='output_activation_function',
        type=str,
        choices=['sigmoid', 'linear', 'sine'],
        help='Activation function of output nodes in the neural networks',
        required=False,
        default="sigmoid"
    )
    arg_parser.add_argument(
        '--fitness',
        dest='fitness',
        type=str,
        help='similarity: Average local similarity, calculated with euclidean distance between'
             ' feature vectors for each frame. multi-objective optimizes for a diverse'
             ' population that consists of various non-dominated trade-offs between similarity'
             ' in different features. Hybrid fitness is the sum of similarity and multi-objective,'
             ' and gives you the best of both worlds. Novelty fitness ignores the objective and'
             ' optimizes for novelty. Mixed fitness chooses a random fitness evaluator for each'
             ' generation.',
        choices=['similarity', 'multi-objective', 'hybrid', 'novelty', 'mixed'],
        required=False,
        default="similarity"
    )
    arg_parser.add_argument(
        '--effect',
        '--effects',
        dest='effect_names',
        type=str,
        nargs='+',
        help='The name(s) of the sound effect(s) to use. See the effects folder for options. In'
             ' composite effects, use "new_layer" to separate layers of parallel effects.',
        required=False,
        default=["dist_lpf"]
    )
    arg_parser.add_argument(
        '--experiment-settings',
        dest='experiment_settings',
        type=str,
        help='Filename of json file in the experiment_settings folder. This file specifies which'
             ' features to use as neural input and for similarity calculations.',
        required=False,
        default="mfcc_basic.json"
    )
    arg_parser.add_argument(
        '--num-runs',
        dest='num_runs',
        help='Number of times to run the experiment (makes sense if seed is not specified)',
        type=positive_int,
        required=False,
        default=1
    )
    arg_parser.add_argument(
        '--write-stats-every',
        dest='write_stats_every',
        help='Write stats.json every nth generation. Set it to 10 or something larger if you'
             ' plan to run the experiment for hundreds of generations',
        type=positive_int,
        required=False,
        default=1
    )
    args = arg_parser.parse_args()

    settings.VERBOSE = args.verbose

    if args.keep_k_best > args.population_size:
        args.keep_k_best = args.population_size

    if args.elitism < 0.0 or args.elitism > 0.4:
        # MultiNEAT (?) may crash with elitism = 0.5, for some unknown reason
        raise Exception('elitism should be in the range [0.0, 0.4]')

    if args.population_size < 3:
        raise Exception('population size should be at least 3')

    if args.weight_mutation_max_power > args.max_weight / 4.0:
        print('Warning: weight_mutation_max_power is relatively high')

    experiment.Experiment.load_experiment_settings(args.experiment_settings)
    analyze.Analyzer.init_features_list()

    if args.fitness in ['multi-objective', 'hybrid'] and \
                    args.population_size < 2 * len(experiment.Experiment.SIMILARITY_CHANNELS):
        print(
            'Warning: Population size is small. The current experiment has {0}'
            ' similarity channels. \nThe population size should be 2-4 times the number of'
            ' similarity channels in experiments with multi-objective optimization'.format(
                len(experiment.Experiment.SIMILARITY_CHANNELS)
            )
        )

    num_runs = args.num_runs
    del args.num_runs
    for i in range(num_runs):
        if num_runs > 1:
            print('')
            print('------')
            print('Run {}'.format(i + 1))
            print('------')
        neuroevolution.Neuroevolution(args)
