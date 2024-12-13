from __future__ import absolute_import
import unittest
import settings
import sound_file
import cross_adapt
import time
import project
import effect
import copy
import os
import experiment
import analyze


class TestCrossAdapt(unittest.TestCase):
    def setUp(self):
        settings.INPUT_DIRECTORY = 'test_audio'
        self.files_to_delete = []
        experiment.Experiment.load_experiment_settings('mfcc_basic.json')
        experiment.Experiment.folder_name = 'test'
        experiment.Experiment.ensure_folders()
        analyze.Analyzer.init_features_list()

    def tearDown(self):
        for file_path in self.files_to_delete:
            os.remove(file_path)

    def test_sound_file(self):
        that_effect = effect.Effect('dist_lpf')
        target_sound = sound_file.SoundFile('drums.wav')
        input_sound = sound_file.SoundFile('noise.wav')

        sounds = [target_sound, input_sound]
        project.Project(sounds)

        num_frames = min(
            target_sound.get_num_frames(),
            input_sound.get_num_frames()
        )
        print('num_frames', num_frames)
        constant_parameter_vector = [0.5] * that_effect.num_parameters
        parameter_vectors = [copy.deepcopy(constant_parameter_vector) for _ in range(num_frames)]

        self.start_time = time.time()

        cross_adapter = cross_adapt.CrossAdapter(
            input_sound=input_sound,
            neural_input_vectors=[],
            effect=that_effect
        )

        output_filename = 'test_cross_adapt.wav'

        process, output_sound_file, csd_path = cross_adapter.cross_adapt(
            parameter_vectors,
            that_effect,
            output_filename
        )

        self.files_to_delete.append(csd_path)

        print('process', process)
        print('output file', output_sound_file)
        print('csd path', csd_path)

        process.wait()

        print("Execution time: {0} seconds".format(
            time.time() - self.start_time)
        )


if __name__ == '__main__':
    unittest.main()
