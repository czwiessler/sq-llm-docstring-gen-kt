from __future__ import absolute_import
from __future__ import division
import wave
import contextlib
import os
import settings
import analyze
import six
import experiment


class SoundFile(object):
    def __init__(self, filename, is_input=True, verify_file=False):
        self.is_input = is_input
        self.filename = filename
        if self.is_input:
            self.file_path = os.path.join(settings.INPUT_DIRECTORY, self.filename)
        else:
            self.file_path = os.path.join(
                settings.OUTPUT_DIRECTORY,
                experiment.Experiment.folder_name,
                self.filename
            )

        self.num_samples = None
        self.sample_rate = None
        self.duration = None
        self.num_channels = None

        if verify_file:
            self.verify_file()
        
        self.duration = None
        self.analysis = {
            'ksmps': settings.HOP_SIZE,
            'series': {}
        }
        self.is_silent = False

    def verify_file(self):
        if not os.path.exists(self.file_path):
            raise Exception(
                'Could not find "{}". Make sure it exists and try again.'.format(self.file_path)
            )
        self.calculate_wav_properties()
        if self.num_samples < settings.FRAME_SIZE:
            raise Exception('The sound {} is too short'.format(self.file_path))
        if self.sample_rate != settings.SAMPLE_RATE:
            raise Exception(
                'Sample rate mismatch: The sample rate of {0} is {1} but should be {2}'.format(
                    self.file_path,
                    self.sample_rate,
                    settings.SAMPLE_RATE
                )
            )
        if self.num_channels != 1:
            raise Exception(
                '{0} has {1} channels, but should have 1 (mono)'.format(
                    self.file_path,
                    self.num_channels
                )
            )

    def calculate_wav_properties(self):
        with contextlib.closing(wave.open(self.file_path, 'r')) as f:
            self.num_samples = f.getnframes()  # number of raw samples
            self.sample_rate = f.getframerate()
            self.duration = self.num_samples / self.sample_rate
            self.num_channels = f.getnchannels()

    def get_duration(self):
        if self.duration is None:
            self.calculate_wav_properties()
        return self.duration

    def get_num_frames(self):
        # number of time steps (k rate)
        arbitrary_series = six.next(six.itervalues(self.analysis['series']))
        return len(arbitrary_series)

    def get_standardized_neural_input_vector(self, k):
        """
        Get a neural input vector for a given frame k.
        Assumes that self.analysis['series_standardized'] is defined and k is within bounds.
        May raise an exception otherwise
        :param k:
        :return: list
        """
        feature_vector = []
        for i, feature in enumerate(experiment.Experiment.NEURAL_INPUT_CHANNELS):
            feature_vector.append(self.analysis['series_standardized'][i][k])
        return feature_vector

    def get_serialized_representation(self):
        self.analysis['order'] = analyze.Analyzer.FEATURES_LIST
        return {
            'feature_data': self.analysis,
            'filename': self.filename,
            'is_input': self.is_input
        }

    def delete(self):
        try:
            os.remove(self.file_path)
        except OSError:
            pass
