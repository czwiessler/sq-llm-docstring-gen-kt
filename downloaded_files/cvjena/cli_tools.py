import numpy as np
import datetime
import csv
from maxdiv import maxdiv, preproc

def read_csv_timeseries(input, selected_variables, timecol, timeformat, maxdatapoints):
    print ("Reading the time series")
    X = []
    times = []
    with open(input, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        if not timecol in reader.fieldnames:
            raise Exception("No column with name {} found in the file".format(timecol))
        if selected_variables is None:
            variables = list(reader.fieldnames)
        else:
            variables = selected_variables
        if timecol in variables:
            variables.remove(timecol)
        print ("Variables used: {}".format(variables))

        for row in reader:
            time_string = row[timecol]
            try:
                current_time = datetime.datetime.strptime(time_string, timeformat)
            except:
                raise Exception("Unable to convert the time specification {} using the format {}".format(time_string, timeformat))
            times.append(current_time)
            vector = [ float(row[v]) for v in variables ]
            X.append(vector)

            if not maxdatapoints is None and len(X) >= maxdatapoints:
                break

    X = np.vstack(X).T
    print ("Data points in the time series: {}".format(X.shape[1]))
    print ("Dimensions for each data point: {}".format(X.shape[0]))

    return X, times

def get_algorithm_parameters():
    return ['extint_min_len', 'extint_max_len', 'alpha', 'mode', 'method', 'num_intervals', 'preproc', 'td_dim', 'td_lag', 'proposals',
            'pca_dim', 'random_projection_dim', 'num_hist', 'num_bins', 'discount'] 

def add_algorithm_parameters(parser):
    parser.add_argument('--method', help='maxdiv method', choices=maxdiv.get_available_methods(), required=True)
    parser.add_argument('--kernel_sigma_sq', help='kernel sigma square hyperparameter for Parzen estimation', type=float, default=1.0)
    parser.add_argument('--num_hist', help='The number of histograms used by the ERPH estimator', type=int, default=100)
    parser.add_argument('--num_bins', help='The number of bins in the histograms used by the ERPH estimator (0 = auto)', type=int, default=0)
    parser.add_argument('--discount', help='Discount added to all bins of the histograms of the ERPH estimator in order to make unseen values not completely unlikely', type=float, default=1)
    parser.add_argument('--extint_min_len', help='minimum length of the extreme interval', default=20, type=int)
    parser.add_argument('--extint_max_len', help='maximum length of the extreme interval', default=100, type=int)
    parser.add_argument('--alpha', help='Hyperparameter for the KL divergence', type=float, default=1.0)
    parser.add_argument('--mode', help='Mode for KL divergence computation', choices=['OMEGA_I', 'SYM', 'I_OMEGA', 'TS', 'LAMBDA', 'IS_I_OMEGA', 'JSD', 'CROSSENT', 'CROSSENT_TS'], default='I_OMEGA')
    parser.add_argument('--num_intervals', help='number of intervals to be displayed', default=0, type=int)
    parser.add_argument('--preproc', help='use a pre-processing method', default=None, choices=preproc.get_available_methods())
    parser.add_argument('--td_dim', help='Time-Delay Embedding Dimension (may be set to 0 for automatic determination)', default=1, type=int)
    parser.add_argument('--td_lag', help='Time-Lag for Time-Delay Embedding (may be set to 0 for automatic determination)', default=1, type=int)
    parser.add_argument('--pca_dim', help='Reduce data to the given number of dimensions using PCA', default=0, type=int)
    parser.add_argument('--random_projection_dim', help='Project data onto the given number of random projection vectors', default=0, type=int)
    parser.add_argument('--proposals', help='method for interval proposing', default='dense', choices=['dense','hotellings_t','kde'])
    parser.add_argument('--prop_th', help='threshold for pointwise interval proposing', type=float, default=1.5)
    parser.add_argument('--prop_mad', help='use MAD to determine the threshold for interval proposing', action='store_true')
    parser.add_argument('--prop_unfiltered', help='use pointwise scores directly for proposals instead of their gradient', action='store_true')
 
