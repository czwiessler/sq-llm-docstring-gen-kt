
import unittest

from hmm_evolve import HMMParams
from ann_evolve import ANNParams
from fsm_evolve import FSMParams
from lookup_evolve import LookerUpParams
from evolve_utils import load_params


class TestRepr(unittest.TestCase):

    def test_repr_load(self):
        for params_class, filename in [
            (ANNParams, "ann_params.csv"),
            (LookerUpParams, "lookup_params.csv"),
            (FSMParams, "fsm_params.csv"),
            (HMMParams, "hmm_params.csv")
        ]:
            print(filename)
            params = load_params(params_class, filename, 10)

