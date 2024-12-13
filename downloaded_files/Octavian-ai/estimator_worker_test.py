
import unittest
import os
import os.path
import sys
import pathlib

from .helpers import *
from .args import get_args
from pbt import EstimatorWorker, WorkerTestCase

args = get_args()

class EstimatorWorkerTestCase(WorkerTestCase):
	
	def vend_worker(self):
		init_params = gen_worker_init_params(args)
		param_spec = gen_param_spec(args)
		params = param_spec.realize()
		print(params)

		return EstimatorWorker(init_params, params)

	def load_worker(self, file_path):
		init_params = gen_worker_init_params(args)
		return EstimatorWorker.load(file_path, init_params)



if __name__ == '__main__':
	install_logging(args)
	unittest.main(module='experiment',defaultTest="EstimatorWorkerTestCase")
