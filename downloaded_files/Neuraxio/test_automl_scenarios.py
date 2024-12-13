
import copy
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple, Type

import numpy as np
import pytest
from neuraxle.base import CX, BaseStep, HandleOnlyMixin, Identity, MetaStep, NonFittableMixin, TrialStatus
from neuraxle.data_container import DataContainer as DACT
from neuraxle.data_container import PredsDACT, TrainDACT
from neuraxle.distributed.streaming import ParallelQueuedFeatureUnion
from neuraxle.hyperparams.distributions import DiscreteHyperparameterDistribution, PriorityChoice, RandInt, Uniform
from neuraxle.hyperparams.space import FlatDict, HyperparameterSamples, HyperparameterSpace
from neuraxle.metaopt.auto_ml import ControlledAutoML, DefaultLoop, Trainer
from neuraxle.metaopt.callbacks import MetricCallback
from neuraxle.metaopt.context import AutoMLContext
from neuraxle.metaopt.data.aggregates import Round
from neuraxle.metaopt.data.reporting import RoundReport
from neuraxle.metaopt.data.vanilla import BaseDataclass, RoundDataclass, ScopedLocation
from neuraxle.metaopt.optimizer import GridExplorationSampler
from neuraxle.metaopt.repositories.db import SQLLiteHyperparamsRepository
from neuraxle.metaopt.repositories.json import HyperparamsOnDiskRepository
from neuraxle.metaopt.repositories.repo import HyperparamsRepository, VanillaHyperparamsRepository
from neuraxle.metaopt.validation import ValidationSplitter
from neuraxle.pipeline import Pipeline
from neuraxle.steps.data import DataShuffler
from neuraxle.steps.flow import TrainOnlyWrapper
from neuraxle.steps.misc import AssertFalseStep, Sleep
from neuraxle.steps.numpy import AddN
from sklearn.metrics import median_absolute_error


class StepThatAssertsContextIsSpecifiedAtTrain(Identity):
    def __init__(self, expected_loc: ScopedLocation, up_to_dc: Type[BaseDataclass] = RoundDataclass):
        BaseStep.__init__(self)
        HandleOnlyMixin.__init__(self)
        self.expected_loc = expected_loc
        self.up_to_dc: Type[BaseDataclass] = up_to_dc

    def _did_process(self, data_container: DACT, context: CX) -> DACT:
        if self.is_train:
            context: AutoMLContext = context  # typing annotation for IDE
            self._assert_equals(
                self.expected_loc[:self.up_to_dc], context.loc[:self.up_to_dc],
                f'Context is not at the expected location. '
                f'Expected {self.expected_loc}, got {context.loc}.',
                context)
            self._assert_equals(
                context.loc in context.repo.wrapped.root, True,
                "Context should have the dataclass, but it doesn't", context)
        return data_container


def test_automl_context_is_correctly_specified_into_trial_with_full_automl_scenario(tmpdir):
    # This is a large test
    dact = DACT(di=list(range(10)), eo=list(range(10, 20)))
    cx = AutoMLContext.from_context(CX(root=tmpdir))
    expected_deep_cx_loc = ScopedLocation.default(0, 0, 0)
    assertion_step = StepThatAssertsContextIsSpecifiedAtTrain(expected_loc=expected_deep_cx_loc)
    automl: ControlledAutoML[Pipeline] = _create_automl_test_loop(tmpdir, assertion_step)
    automl = automl.handle_fit(dact, cx)

    pred: DACT = automl.handle_predict(dact.without_eo(), cx)

    best: Tuple[float, int, FlatDict] = automl.report.best_result_summary()
    best_score = best[0]
    assert best_score == 0
    best_add_n: int = list(best[-1].values())[0]
    assert best_add_n == 10  # We expect the AddN step to make use of the value "10" for 0 MAE error.
    assert np.array_equal(list(pred.di), list(dact.eo))


def test_automl_step_can_interrupt_on_fail_with_full_automl_scenario(tmpdir):
    # This is a large test
    dact = DACT(di=list(range(10)), eo=list(range(10, 20)))
    cx = CX(root=tmpdir)
    assertion_step = AssertFalseStep()
    automl = _create_automl_test_loop(tmpdir, assertion_step)

    with pytest.raises(AssertionError):
        automl.handle_fit(dact, cx)


def _create_automl_test_loop(tmpdir, assertion_step: BaseStep, n_trials: int = 4, start_new_round=True, refit_best_trial=True):
    automl = ControlledAutoML(
        pipeline=Pipeline([
            TrainOnlyWrapper(DataShuffler()),
            AddN().with_hp_range(range(8, 12)),
            assertion_step
        ]),
        loop=DefaultLoop(
            trainer=Trainer(
                validation_splitter=ValidationSplitter(validation_size=0.2),
                n_epochs=4,
                callbacks=[MetricCallback('MAE', median_absolute_error, higher_score_is_better=False)],
            ),
            hp_optimizer=GridExplorationSampler(n_trials),
            n_trials=n_trials,
            continue_loop_on_error=False,
        ),
        main_metric_name='MAE',
        start_new_round=start_new_round,
        refit_best_trial=refit_best_trial,
    )

    return automl


@pytest.mark.parametrize('n_trials', [1, 3, 4, 8, 12, 13, 16, 17, 20])
def test_grid_sampler_fulls_grid(n_trials):
    _round, hp_space, ges = _get_optimization_scenario(n_trials)

    tried: Set[Dict[str, Any]] = set()
    for i in range(n_trials):
        hp = ges.find_next_best_hyperparams(_round, hp_space).to_flat_dict()
        _round.add_testing_optim_result(hp)
        flat = frozenset(hp.items())

        assert flat not in tried, f"Found the same hyperparameter samples twice: `{flat}`, with {i+1} past trials."
        tried.add(flat)
    assert len(tried) == n_trials


@pytest.mark.parametrize('n_trials', [1, 3, 4, 12, 13, 16, 17, 20, 40, 50, 100])
def test_grid_sampler_fulls_individual_params(n_trials):
    _round, hp_space, ges = _get_optimization_scenario(n_trials=n_trials)
    ges: GridExplorationSampler = ges  # typing
    _round: Round = _round  # typing

    tried_params: Dict[str, Set[Any]] = defaultdict(set)
    for i in range(n_trials):
        hp = ges.find_next_best_hyperparams(_round, hp_space).to_flat_dict()
        _round.add_testing_optim_result(hp)
        for hp_k, value_set in hp.items():
            tried_params[hp_k].add(value_set)

    for hp_k, value_set in tried_params.items():
        hp_k: str = hp_k  # typing
        value_set: Set[Any] = value_set  # typing

        if isinstance(hp_space[hp_k], DiscreteHyperparameterDistribution):
            round_numbs = min(n_trials, len(hp_space[hp_k].values()))
            assert len(value_set) == round_numbs
        else:
            round_numbs = min(n_trials, len(ges.flat_hp_grid_values[hp_k]))
            assert len(value_set) >= round_numbs, (
                f"value_set={value_set} has a len={len(value_set)}, but len={round_numbs} was expected.")


@dataclass
class RoundReportStub:
    _all_tried_hyperparams: List[FlatDict] = field(default_factory=list)

    def add_testing_optim_result(self, rdict: HyperparameterSamples):
        self._all_tried_hyperparams.append(rdict)

    def get_all_hyperparams(self):
        return self._all_tried_hyperparams


def _get_optimization_scenario(n_trials):
    hp_space = HyperparameterSpace({
        'a__add_n': Uniform(0, 4),
        'b__multiply_n': RandInt(0, 4),
        'c__Pchoice': PriorityChoice(["one", "two"]),
    })
    _round: RoundReport = RoundReportStub()
    ges = GridExplorationSampler(n_trials)
    return _round, hp_space, ges


class OnlyFitAtTransformTime(NonFittableMixin, MetaStep):
    """
    This is needed for AutoML to make sure that fitting will happen in parallel, not in the same thread.
    Note that nothing is returned, the step does not transform the data. It only fits.
    """

    def _transform_data_container(self, data_container: TrainDACT, context: CX) -> PredsDACT:
        self.wrapped = self.wrapped.handle_fit(data_container, context)
        return data_container


def test_automl_can_resume_last_run_and_retrain_best_with_0_trials(tmpdir):
    dact = DACT(di=list(range(10)), eo=list(range(10, 20)))
    cx = AutoMLContext.from_context(CX(root=tmpdir))
    sleep_step = Sleep(0.001)
    n_trials = 4
    automl: ControlledAutoML = _create_automl_test_loop(
        tmpdir, sleep_step, n_trials=1, start_new_round=False, refit_best_trial=False
    )
    for _ in range(n_trials):
        copy.deepcopy(automl).handle_fit(dact, cx)

    automl_refiting_best: ControlledAutoML = _create_automl_test_loop(
        tmpdir, sleep_step, n_trials=0, start_new_round=False, refit_best_trial=True)
    automl_refiting_best, preds = automl_refiting_best.handle_fit_transform(dact, cx)

    bests: List[Tuple[TrialStatus, float, int, FlatDict]] = automl_refiting_best.report.summary()
    hps: List[FlatDict] = automl_refiting_best.report.get_all_hyperparams()

    assert len(bests) == n_trials
    assert len(set(tuple(hp.items()) for hp in hps)) == n_trials
    assert len(set(tuple(dict(hp).items()) for score, i, status, hp in bests)
               ) == n_trials, f"Expecting unique hyperparams for the given n_trials={n_trials} and intelligent grid sampler."

    best_score = bests[0][0]
    assert median_absolute_error(dact.eo, preds.di) == best_score


@pytest.mark.parametrize("use_processes,repoclass", [
    [False, VanillaHyperparamsRepository],
    [False, HyperparamsOnDiskRepository],
    [True, HyperparamsOnDiskRepository],
    # TODO: 'SQLite objects created in a thread can only be used in that same thread.' would need a pack and unpack service method upon pre and post threading.
    # [True, SQLLiteHyperparamsRepository],
    # [False, SQLLiteHyperparamsRepository],
])
def test_automl_use_a_json_repo_in_parallelized_round(use_processes, repoclass: Type[HyperparamsRepository]):
    for _ in range(1):  # Editable range for debugging if a flickering corner-case is rare.
        tmpdir = CX.get_new_cache_folder()
        len_x = 2 * 3 * 4 * 5
        dact = DACT(di=list(range(len_x)), eo=list(range(10, 10 + len_x)))
        repo = repoclass(tmpdir).with_lock()
        cx = AutoMLContext.from_context(repo=repo)
        sleep_step = Sleep(0.125, add_random_quantity=0.250)
        automl: ControlledAutoML = _create_automl_test_loop(
            tmpdir, sleep_step, n_trials=1, start_new_round=False, refit_best_trial=False
        )
        n_sequential_steps = 3
        n_workers_in_parallel_per_step = 4  # TODO: use something else, such as 2, 3 or 4.
        n_minibatches_in_series = 5
        n_trials = n_sequential_steps * n_workers_in_parallel_per_step * n_minibatches_in_series
        parallel_automl = ParallelQueuedFeatureUnion(
            # steps=[OnlyFitAtTransformTime(automl)],
            steps=[
                # TODO: this test seems broken because it never creates the first rounds - always wants to extend it.
                OnlyFitAtTransformTime(copy.deepcopy(automl))
                for _ in range(n_sequential_steps)
            ],
            batch_size=int(len(dact) / n_minibatches_in_series / n_workers_in_parallel_per_step),
            n_workers_per_step=n_workers_in_parallel_per_step,
            use_processes=use_processes,
            use_savers=False,
            max_queued_minibatches=1  # TODO: why 1 here?
        )
        parallel_automl.handle_fit_transform(dact, cx)
        # for i in range(n_trials):
        #     OnlyFitAtTransformTime(copy.deepcopy(automl)).handle_fit_transform(dact, cx)

        # automl_refiting_best: ControlledAutoML = _create_automl_test_loop(
        #     tmpdir, sleep_step, n_trials=0, start_new_round=False, refit_best_trial=True)
        automl_refiting_best = automl.to_force_refit_best_trial()
        automl_refiting_best, preds = automl_refiting_best.handle_fit_transform(dact, cx)

        bests: List[Tuple[float, int, TrialStatus, FlatDict]] = automl_refiting_best.report.summary()

        assert len(bests) == n_trials

        best_score = bests[0][0]
        assert median_absolute_error(dact.eo, preds.di) == best_score
        assert 0.0 == best_score
