from __future__ import absolute_import
import unittest
import settings
import sound_file
import template_handler
import csound_handler
import os
import time
import experiment


class TestCsound(unittest.TestCase):
    def setUp(self):
        settings.INPUT_DIRECTORY = 'test_audio'
        self.num_sounds = 10
        self.drums = sound_file.SoundFile('drums.wav')
        self.files_to_delete = []
        self.template_file_path = os.path.join(settings.EFFECT_DIRECTORY, 'test_effect.csd.jinja2')
        with open(self.template_file_path, 'r') as template_file:
            self.template_string = template_file.read()
        experiment.Experiment.folder_name = 'test'

    def tearDown(self):
        for file_path in self.files_to_delete:
            os.remove(file_path)

    def test_serial_execution(self):
        self.start_time = time.time()
        template = template_handler.TemplateHandler(settings.EFFECT_DIRECTORY, self.template_string)
        template.compile(
            ksmps=settings.HOP_SIZE,
            duration=self.drums.get_duration()
        )

        for i in range(self.num_sounds):
            csd_path = os.path.join(
                settings.CSD_DIRECTORY,
                experiment.Experiment.folder_name,
                'test_effect_{}.csd'.format(i)
            )
            template.write_result(csd_path)
            csound = csound_handler.CsoundHandler(csd_path)
            output_filename = self.drums.filename + '.test_processed_serial_{}.wav'.format(i)
            output_file_path = os.path.join(
                settings.OUTPUT_DIRECTORY,
                experiment.Experiment.folder_name,
                output_filename
            )
            csound.run(
                input_file_path=self.drums.file_path,
                output_file_path=output_file_path,
                async=False
            )
            self.files_to_delete.append(csd_path)
            self.files_to_delete.append(output_file_path)

        print("Serial execution time for {0} sounds: {1} seconds".format(
            self.num_sounds,
            time.time() - self.start_time)
        )

    def test_parallel_execution(self):
        self.start_time = time.time()

        template = template_handler.TemplateHandler(settings.EFFECT_DIRECTORY, self.template_string)
        template.compile(
            ksmps=settings.HOP_SIZE,
            duration=self.drums.get_duration()
        )

        processes = []

        for i in range(self.num_sounds):
            csd_path = os.path.join(
                settings.CSD_DIRECTORY,
                experiment.Experiment.folder_name,
                'test_effect_{}.csd'.format(i)
            )
            template.write_result(csd_path)
            csound = csound_handler.CsoundHandler(csd_path)
            output_filename = self.drums.filename + '.test_processed_parallel_{}.wav'.format(i)
            output_file_path = os.path.join(
                settings.OUTPUT_DIRECTORY,
                experiment.Experiment.folder_name,
                output_filename
            )
            p = csound.run(
                input_file_path=self.drums.file_path,
                output_file_path=output_file_path,
                async=False
            )
            processes.append(p)
            self.files_to_delete.append(csd_path)
            self.files_to_delete.append(output_file_path)

        for p in processes:
            p.wait()

        print("Parallel execution time for {0} sounds: {1} seconds".format(
            self.num_sounds,
            time.time() - self.start_time)
        )


if __name__ == '__main__':
    unittest.main()
