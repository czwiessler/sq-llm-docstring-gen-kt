#      Copyright (c) 2019  Lukasz Tracewski
#
#      This file is part of Audio Explorer.
#
#      Audio Explorer is free software: you can redistribute it and/or modify
#      it under the terms of the GNU General Public License as published by
#      the Free Software Foundation, either version 3 of the License, or
#      (at your option) any later version.
#
#      Audio Explorer is distributed in the hope that it will be useful,
#      but WITHOUT ANY WARRANTY; without even the implied warranty of
#      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#      GNU General Public License for more details.
#
#      You should have received a copy of the GNU General Public License
#      along with Audio Explorer.  If not, see <https://www.gnu.org/licenses/>.

import aubio
import numpy as np
import pandas as pd


def get_pitch_stats(signal: np.ndarray, fs: int, block_size: int, hop: int, tolerance: float = 0.8,
                    algorithm = 'yinfft') -> dict:
    """
    Get basic statistic on pitch in the given signal
    :param signal: 1-d signal
    :param fs: sampling frequency
    :param block_size: window size
    :param hop: size of a hop between frames
    :param tolerance:  tolerance for the pitch detection algorithm (for aubio)
    :return:
    """
    pitch_o = aubio.pitch(algorithm, block_size, hop, fs)
    pitch_o.set_unit('Hz')
    pitch_o.set_tolerance(tolerance)
    signal_win = np.array_split(signal, np.arange(hop, len(signal), hop))

    pitch_array = []
    for frame in signal_win[:-1]:
        pitch = pitch_o(frame)[0]
        if pitch > 0:
            pitch_array.append(pitch)

    if pitch_array:
        pitch_array = np.array(pitch_array)
        Q25, Q50, Q75 = np.quantile(pitch_array, [0.25, 0.50, 0.75])
        IQR = Q75 - Q25
        median = np.median(pitch_array)
        pitch_min = pitch_array.min()
        pitch_max = pitch_array.max()
    else:
        Q25 = 0
        Q50 = 0
        Q75 = 0
        median = 0
        IQR = 0
        pitch_min = 0
        pitch_max = 0

    pitchstats = {
        'pitch_median': median,
        'pitch_mean': Q50,
        'pitch_Q25': Q25,
        'pitch_Q75': Q75,
        'pitch_IQR': IQR,
        'pitch_min': pitch_min,
        'pitch_max': pitch_max
    }

    return pitchstats


def get_pitch_stats_series(signal: np.ndarray, fs: int, block_size: int, hop: int, tolerance: float = 0.5) -> pd.Series:
    """
    Get basic statistic on pitch in the given signal
    :param signal: 1-d signal
    :param fs: sampling frequency
    :param block_size: window size
    :param hop: size of a hop between frames
    :param tolerance:  tolerance for the pitch detection algorithm (for aubio)
    :return:
    """
    pitchstats = get_pitch_stats(signal, fs, block_size, hop, tolerance)
    return pd.Series(pitchstats)

