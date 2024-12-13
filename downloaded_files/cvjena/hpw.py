""" Runs the MDI algorithm on the Hurricane time-series. """

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pylab as plt
import csv, datetime
from collections import OrderedDict

from maxdiv import maxdiv, preproc, baselines_noninterval, eval


HURRICANE_GT = { \
    'Sandy'     : (datetime.date(2012,10,22), datetime.date(2012,10,29)),
    'Rafael'    : (datetime.date(2012,10,12), datetime.date(2012,10,18)),
    'Isaac'     : (datetime.date(2012, 8,22), datetime.date(2012, 8,25))
}


def read_hpw_csv(csvFile):
    """ Reads HPW data from a CSV file.
    
    The CSV file must contain 4 fields per line:
    date as 'yyyy-m-d-h', wind speed, air pressure and wave height
    The first line is assumed to be field headings and will be ignored.
    """
    
    ts = []
    dates = []
    mask = []
    with open(csvFile) as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            if i == 0:
                continue
            fields = [float(x) for x in line[1:]]
            if np.isnan(fields).any():
                ts.append([0] * len(fields))
                mask.append(True)
            else:
                ts.append(fields)
                mask.append(False)
            dates.append(datetime.datetime(*[int(x) for i, x in enumerate(line[0].split('-'))]))
    
    return np.ma.MaskedArray(np.array(ts).T, np.array(mask).reshape(1, len(mask)).repeat(3, axis = 0)), dates


def read_reduced_hpw_csv(csvFile):
    """ Reads HPW data from a CSV file and takes 4-hourly mean values.
    
    The CSV file must contain 4 fields per line:
    date as 'yyyy-m-d-h', wind speed, air pressure and wave height
    The first line is assumed to be field headings and will be ignored.
    """
    
    # Read data from CSV file into ordered dict
    data = OrderedDict()
    with open(csvFile) as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            if i == 0:
                continue
            date = tuple(int(x) if i < 3 else int(int(x) / 4) * 4 for i, x in enumerate(line[0].split('-')))
            fields = [float(x) for x in line[1:]]
            if not np.any(np.isnan(fields)):
                if date not in data:
                    data[date] = []
                data[date].append(fields)
    
    # Take 4-hourly means and store them in a numpy array
    ts = np.ndarray((3, len(data)))
    for i, (date, values) in enumerate(data.items()):
        ts[:,i] = np.array(values).mean(axis = 0).T
    dates = [datetime.datetime(*date) for date in data.keys()]
    
    return ts, dates


def datetime_diff(a, b):
    """ Calculates the difference a - b between to dates in hours. """
    
    if isinstance(a, datetime.date):
        a = datetime.datetime.combine(a, datetime.datetime.min.time())
    if isinstance(b, datetime.date):
        b = datetime.datetime.combine(b, datetime.datetime.min.time())
    
    return int((a-b).total_seconds()) / 3600


if __name__ == '__main__':

    import sys
    method = sys.argv[1] if len(sys.argv) > 1 else 'gaussian_cov_ts'
    propmeth = sys.argv[2] if len(sys.argv) > 2 else 'dense'

    # Load data
    data, dates = read_hpw_csv('HPW_2012_41046.csv')
    data = preproc.normalize_time_series(data)
    
    # Detect
    if method in ['hotellings_t', 'kde']:
        if method == 'kde':
            scores = baselines_noninterval.pointwiseKDE(preproc.td(data))
        else:
            scores = baselines_noninterval.hotellings_t(preproc.td(data))
        regions = baselines_noninterval.pointwiseScoresToIntervals(scores, 24)
    elif method == 'gaussian_cov_ts':
        regions = maxdiv.maxdiv(data, 'gaussian_cov', mode = 'TS', td_dim = 3, td_lag = 1, proposals = propmeth,
                                extint_min_len = 24, extint_max_len = 72, num_intervals = 5)
    else:
        regions = maxdiv.maxdiv(data, method, mode = 'I_OMEGA', td_dim = 3, td_lag = 1, proposals = propmeth,
                                extint_min_len = 24, extint_max_len = 72, num_intervals = 5)
    
    # Console output
    print('-- Ground Truth --')
    for name, (a, b) in HURRICANE_GT.items():
        print('{:{}s}: {!s} - {!s}'.format(name, max(len(n) for n in HURRICANE_GT.keys()), a, b - datetime.timedelta(days = 1)))
    print('\n-- Detected Intervals ({} with {} proposals) --'.format(method, propmeth))
    for a, b, score in regions:
        print('{!s} - {!s} (Score: {})'.format(dates[a], dates[b-1], score))
    
    # Plot
    ygt = [(datetime_diff(a, dates[0]), datetime_diff(b, dates[0])) for a, b in HURRICANE_GT.values()]
    eval.plotDetections(data, regions, ygt,
                        ticks = { datetime_diff(d, dates[0]) : d.strftime('%b %Y') for d in (datetime.date(2012,mon,1) for mon in range(6, 12)) })