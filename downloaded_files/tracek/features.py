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

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, cpu_count
from audioexplorer import specprop, pitchprop, melprop
from audioexplorer.onsets import OnsetDetector
from audioexplorer.filters import frequency_filter
from audioexplorer.yaafe_wrapper import YaafeWrapper, YAAFE_FEATURES


FEATURES = {'freq': 'Frequency statistics',
            'pitch': 'Pitch statistics'}
FEATURES.update(YAAFE_FEATURES)


class FeatureExtractor(object):

    def __init__(self, fs: int, block_size: int=512, step_size: int=None, selected_features='all'):
        self.fs = fs
        self.block_size = block_size
        if selected_features == 'all':
            self.selected_features = list(FEATURES.keys())
        else:
            self.selected_features = selected_features
        if not step_size:
            self.step_size = block_size // 2

        any_yaafe_feature = set(self.selected_features) & set(YAAFE_FEATURES.keys())
        if not any_yaafe_feature:
            self.yaafe = None
        else:
            self.yaafe = YaafeWrapper(fs, block_size, step_size, selected_features=any_yaafe_feature)


    def get_features(self, sample: np.ndarray) -> pd.DataFrame:
        computed_features = []
        if 'freq' in self.selected_features:
            spectral_props = specprop.spectral_statistics_series(sample, self.fs)
            computed_features.append(spectral_props)
        if 'pitch' in self.selected_features:
            hop = self.step_size // 2
            pitch_stats = pitchprop.get_pitch_stats_series(sample, self.fs, block_size=self.block_size, hop=hop,
                                                           tolerance=0.4)
            computed_features.append(pitch_stats)
        if self.yaafe:
            yaafe_features = self.yaafe.get_mean_features_as_series(sample)
            computed_features.append(yaafe_features)
        r = pd.concat(computed_features)
        return r


def _extract_features(samples: np.ndarray, fs: int, block_size: int, selected_features):
    extractor = FeatureExtractor(fs=fs, block_size=block_size, selected_features=selected_features)
    features = []
    for sample in samples:
        f = extractor.get_features(sample)
        features.append(f)
    return pd.DataFrame(features)


def _split_audio_into_chunks_by_onsets(X: np.ndarray, fs: int, onsets: np.ndarray, sample_len: float, split: int) -> np.ndarray:
    samples = []
    for onset in onsets:
        start = int(onset * fs)
        end = int((onset + sample_len) * fs)
        samples.append(X[start: end])
    samples = np.array(samples)
    if split == -1:
        split = cpu_count()
    if split > 1:
        samples = np.array_split(samples, split)
    return samples


def get(X, fs: int, n_jobs: int=1, selected_features='all', **params) -> pd.DataFrame:
    lowcut = int(params.get('lowcut'))
    highcut = int(params.get('highcut'))
    block_size = int(params.get('block_size'))
    step_size = int(params.get('step_size', block_size // 2))
    onset_detector_type = params.get('onset_detector_type')
    onset_threshold = float(params.get('onset_threshold'))
    onset_silence_threshold = float(params.get('onset_silence_threshold'))
    min_duration_s = float(params.get('min_duration_s'))
    sample_len = float(params.get('sample_len'))

    X = frequency_filter(X, fs, lowcut=lowcut, highcut=highcut)

    if onset_threshold > 0:
        onset_detector = OnsetDetector(fs, nfft=block_size, hop=step_size,
                                       onset_detector_type=onset_detector_type,
                                       onset_threshold=onset_threshold, onset_silence_threshold=onset_silence_threshold,
                                       min_duration_s=min_duration_s)
        onsets = onset_detector.get_all(X)
    else:
        onsets = np.arange(0, len(X) / fs, sample_len)
    chunks = _split_audio_into_chunks_by_onsets(X, fs, onsets, sample_len, n_jobs)
    if n_jobs == 1:
        features = _extract_features(chunks, fs, block_size=block_size, selected_features=selected_features)
    else:
        features = Parallel(n_jobs=n_jobs, backend='multiprocessing')(
            delayed(_extract_features)(samples=chunk, fs=fs, block_size=block_size, selected_features=selected_features)
            for chunk in chunks)
        features = pd.concat(features)

    features.insert(0, column='onset', value=onsets)
    features.insert(1, column='offset', value=onsets + sample_len)
    features = features.reset_index(drop=True)
    return features

