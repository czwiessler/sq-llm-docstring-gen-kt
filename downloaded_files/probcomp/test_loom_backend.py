# -*- coding: utf-8 -*-

#   Copyright (c) 2010-2016, MIT Probabilistic Computing Project
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import contextlib
import os
import shutil
import string
import tempfile

import pytest

from bayeslite import bayesdb_open
from bayeslite import bayesdb_register_backend
from bayeslite.core import bayesdb_get_generator
from bayeslite.core import bayesdb_get_population
from bayeslite.exception import BQLError

try:
    from bayeslite.backends.loom_backend import LoomBackend
except ImportError:
    pytest.skip('Failed to import Loom.')


PREDICT_RUNS = 100
X_MIN, Y_MIN = 0, 0
X_MAX, Y_MAX = 200, 100


@contextlib.contextmanager
def tempdir(prefix):
    path = tempfile.mkdtemp(prefix=prefix)
    try:
        yield path
    finally:
        if os.path.isdir(path):
            shutil.rmtree(path)


def test_loom_one_numeric():
    """Simple test of the LoomBackend on a one variable table
    Only checks for errors from the Loom system."""
    from datetime import datetime
    with tempdir('bayeslite-loom') as loom_store_path:
        with bayesdb_open(':memory:') as bdb:
            bayesdb_register_backend(bdb,
                LoomBackend(loom_store_path=loom_store_path))
            bdb.sql_execute('create table t(x)')
            for x in xrange(10):
                bdb.sql_execute('insert into t (x) values (?)', (x,))
            bdb.execute('create population p for t (x numerical)')
            bdb.execute('create generator g for p using loom')
            bdb.execute('initialize 1 models for g')
            bdb.execute('analyze g for 10 iterations')
            bdb.execute('''
                    estimate probability density of x = 50 from p
            ''').fetchall()
            bdb.execute('simulate x from p limit 1').fetchall()
            bdb.execute('drop models from g')
            bdb.execute('drop generator g')
            bdb.execute('drop population p')
            bdb.execute('drop table t')


def test_loom_complex_add_analyze_drop_sequence():
    with tempdir('bayeslite-loom') as loom_store_path:
        with bayesdb_open(':memory:') as bdb:
            bayesdb_register_backend(bdb,
                LoomBackend(loom_store_path=loom_store_path))
            bdb.sql_execute('create table t (x)')
            for x in xrange(10):
                bdb.sql_execute('insert into t (x) values (?)', (x,))
            bdb.execute('create population p for t (x numerical)')
            bdb.execute('create generator g for p using loom')

            bdb.execute('initialize 2 models for g')

            bdb.execute('initialize 3 models if not exists for g')
            population_id = bayesdb_get_population(bdb, 'p')
            generator_id = bayesdb_get_generator(bdb, population_id, 'g')
            cursor = bdb.sql_execute('''
                SELECT num_models FROM bayesdb_loom_generator_model_info
                    WHERE generator_id=?;
            ''',(generator_id,))
            num_models = cursor.fetchall()[0][0]
            # Make sure that the total number of models is
            # 3 and not 2 + 3 = 5.
            assert num_models == 3

            bdb.execute('analyze g for 10 iterations')
            bdb.execute('estimate probability density of x = 50 from p')

            with pytest.raises(BQLError):
                bdb.execute('drop model 1 from g')
            bdb.execute('drop models from g')

            bdb.execute('initialize 1 models for g')
            population_id = bayesdb_get_population(bdb, 'p')
            generator_id = bayesdb_get_generator(bdb, population_id, 'g')
            cursor = bdb.sql_execute('''
                SELECT num_models FROM bayesdb_loom_generator_model_info
                    WHERE generator_id=?;
            ''',(generator_id,))
            num_models = cursor.fetchall()[0][0]
            # Make sure that the number of models was reset after dropping.
            assert num_models == 1
            bdb.execute('analyze g for 50 iterations')

            cursor = bdb.execute('''
                estimate probability density of x = 50 from p''')
            probDensityX1 = cursor.fetchall()
            probDensityX1 = [x[0] for x in probDensityX1]
            bdb.execute('simulate x from p limit 1').fetchall()
            bdb.execute('drop models from g')

            bdb.execute('initialize 1 model for g')
            bdb.execute('analyze g for 50 iterations')
            cursor = bdb.execute('''
                estimate probability density of x = 50 from p''')
            probDensityX2 = cursor.fetchall()
            probDensityX2 = [x[0] for x in probDensityX2]
            # Check that the analysis started fresh after dropping models
            # and that it produces similar results the second time.
            for px1, px2 in zip(probDensityX1, probDensityX2):
                assert abs(px1 - px2) < .01
            bdb.execute('drop models from g')
            bdb.execute('drop generator g')
            bdb.execute('drop population p')
            bdb.execute('drop table t')


def test_stattypes():
    """Test of the LoomBackend on a table with all possible data types.
    Only checks for errors from Loom.
    """
    with tempdir('bayeslite-loom') as loom_store_path:
        with bayesdb_open(':memory:') as bdb:
            bayesdb_register_backend(bdb,
                LoomBackend(loom_store_path=loom_store_path))
            bdb.sql_execute('create table t (u, co, b, ca, cy, nu, no)')
            for _x in xrange(10):
                cat_dict = ['a', 'b', 'c']
                bdb.sql_execute('''
                    insert into t (u, co, b, ca, cy, nu, no)
                    values (?, ?, ?, ?, ?, ?, ?)''',
                    (
                        cat_dict[bdb._prng.weakrandom_uniform(3)],
                        bdb._prng.weakrandom_uniform(200),
                        bdb._prng.weakrandom_uniform(2),
                        cat_dict[bdb._prng.weakrandom_uniform(3)],
                        bdb._prng.weakrandom_uniform(1000)/4.0,
                        bdb._prng.weakrandom_uniform(1000)/4.0 - 100.0,
                        bdb._prng.weakrandom_uniform(1000)/4.0
                    ))
            bdb.execute('''create population p for t(
                u unbounded_nominal;
                co counts;
                b boolean;
                ca nominal;
                cy cyclic;
                nu numerical;
                no nominal)
            ''')
            bdb.execute('create generator g for p using loom')
            bdb.execute('initialize 1 model for g')
            bdb.execute('analyze g for 50 iterations')
            bdb.execute('''estimate probability density of
                (co=2, nu=50, u='a') by p''').fetchall()
            bdb.execute('''estimate probability density of
                (nu = 50, u='a') given (co=2) by p''').fetchall()
            with pytest.raises(Exception):
                # There seems to be an issue with encoding boolean variables
                # in LoomBackend.simulate_joint, although using b=1 in the
                # condition for simulate results in no error.
                bdb.execute('''estimate probability density of
                    (b=0) by p''').fetchall()
            bdb.execute('''simulate u, co, b, ca, cy, nu, no
                from p limit 1''').fetchall()
            bdb.execute('''simulate u, b, ca, no
                from p given nu=3, co=2, b=1 limit 1''').fetchall()
            bdb.execute('drop models from g')
            bdb.execute('drop generator g')
            bdb.execute('drop population p')
            bdb.execute('drop table t')


def test_loom_guess_schema_nominal():
    """Test to make sure that LoomBackend handles the case where the user
    provides a nominal variable with more than 256 distinct values. In this
    case, Loom automatically specifies the unbounded_nominal type.
    """
    with tempdir('bayeslite-loom') as loom_store_path:
        with bayesdb_open(':memory:') as bdb:
            bayesdb_register_backend(bdb,
                LoomBackend(loom_store_path=loom_store_path))
            bdb.sql_execute('create table t (v)')
            vals_to_insert = []
            for i in xrange(300):
                word = ""
                for _j in xrange(20):
                    letter_index = bdb._prng.weakrandom_uniform(
                        len(string.letters))
                    word += string.letters[letter_index]
                vals_to_insert.append(word)
            for i in xrange(len(vals_to_insert)):
                bdb.sql_execute('''
                    insert into t (v) values (?)
                ''', (vals_to_insert[i],))

            bdb.execute('create population p for t (v nominal)')
            bdb.execute('create generator g for p using loom')
            bdb.execute('initialize 1 model for g')
            bdb.execute('analyze g for 50 iterations')
            bdb.execute('drop models from g')
            bdb.execute('drop generator g')
            bdb.execute('drop population p')
            bdb.execute('drop table t')


def test_loom_four_var():
    """Test Loom on a four variable table.
    Table consists of:
    * x - a random int between 0 and 200
    * y - a random int between 0 and 100
    * xx - just 2*x
    * z - a nominal variable that has an even
    chance of being 'a' or 'b'

    Queries run and tested include:
    estimate similarity, estimate probability density, simulate,
    estimate mutual information, estimate dependence probability,
    infer explicit predict
    """
    with tempdir('bayeslite-loom') as loom_store_path:
        with bayesdb_open(':memory:') as bdb:
            bayesdb_register_backend(
                    bdb, LoomBackend(loom_store_path=loom_store_path))
            bdb.sql_execute('create table t(x, xx, y, z)')
            bdb.sql_execute('''
                insert into t (x, xx, y, z) values (100, 200, 50, 'a')''')
            bdb.sql_execute('''
                insert into t (x, xx, y, z) values (100, 200, 50, 'a')''')
            for _index in xrange(100):
                x = bdb._prng.weakrandom_uniform(X_MAX)
                bdb.sql_execute('''
                    insert into t(x, xx, y, z) values(?, ?, ?, ?)
                    ''', (x, x*2, int(bdb._prng.weakrandom_uniform(Y_MAX)),
                        'a' if bdb._prng.weakrandom_uniform(2) == 1
                        else 'b'))

            bdb.execute('''
                create population p for t(x numerical; xx numerical;
                y numerical; z nominal)''')
            bdb.execute('create generator g for p using loom')
            bdb.execute('initialize 10 model for g')
            bdb.execute('analyze g for 20 iterations')

            with pytest.raises(BQLError):
                relevance = bdb.execute('''
                    estimate
                        predictive relevance
                            to hypothetical rows with values ((x=50, xx=100))
                            in the context of "x"
                    from p
                    where rowid = 1
                ''').fetchall()

            relevance = bdb.execute('''
                estimate
                    predictive relevance
                        to existing rows (rowid = 1)
                        in the context of "x"
                from p
                where rowid = 1
            ''').fetchall()
            assert relevance[0][0] == 1

            similarities = bdb.execute('''estimate similarity
                in the context of x from pairwise p limit 2''').fetchall()
            assert similarities[0][2] <= 1
            assert similarities[1][2] <= 1
            assert abs(
                similarities[0][2]-similarities[1][2]) < 0.005

            impossible_density = bdb.execute(
                'estimate probability density of x = %d by p'
                % (X_MAX*2.5,)).fetchall()
            assert impossible_density[0][0] < 0.0001

            possible_density = bdb.execute(
                'estimate probability density of x = %d  by p' %
                ((X_MAX-X_MIN)/2,)).fetchall()
            assert possible_density[0][0] > 0.001

            nominal_density = bdb.execute('''
                estimate probability density of z = 'a' by p
            ''').fetchall()
            assert abs(nominal_density[0][0]-.5) < 0.2

            mutual_info = bdb.execute('''
                estimate mutual information as mutinf
                from pairwise columns of p order by mutinf
            ''').fetchall()
            _, a, b, c = zip(*mutual_info)
            mutual_info_dict = dict(zip(zip(a, b), c))
            assert mutual_info_dict[('x', 'y')] < mutual_info_dict[
                ('x', 'xx')] < mutual_info_dict[('x', 'x')]

            simulated_data = bdb.execute(
                'simulate x, y from p limit %d'
                % (PREDICT_RUNS,)).fetchall()
            xs, ys = zip(*simulated_data)
            assert abs((sum(xs)/len(xs)) - (X_MAX-X_MIN)/2) < \
                    (X_MAX-X_MIN)/5
            assert abs((sum(ys)/len(ys)) - (Y_MAX-Y_MIN)/2) < \
                    (Y_MAX-Y_MIN)/5
            assert sum([1 if (x < Y_MIN or x > X_MAX)
                else 0 for x in xs]) < .5*PREDICT_RUNS
            assert sum([1 if (y < Y_MIN or y > Y_MAX)
                else 0 for y in ys]) < .5*PREDICT_RUNS

            dependence = bdb.execute('''estimate dependence probability
                from pairwise variables of p''').fetchall()
            for (_, col1, col2, d_val) in dependence:
                if col1 == col2:
                    assert d_val == 1
                elif col1 in ['xx', 'x'] and col2 in ['xx', 'x']:
                    assert d_val > 0.80
                else:
                    assert d_val < 0.20
            predict_confidence = bdb.execute(
                'infer explicit predict x confidence x_c FROM p').fetchall()
            predictions, confidences = zip(*predict_confidence)
            assert abs((sum(predictions)/len(predictions))
                - (X_MAX-X_MIN)/2) < (X_MAX-X_MIN)/5
            assert sum([1 if (p < X_MIN or p > X_MAX)
                else 0 for p in predictions]) < .5*PREDICT_RUNS
            assert all([c == 0 for c in confidences])

def test_population_two_generators():
    with tempdir('bayeslite-loom') as loom_store_path:
        with bayesdb_open(':memory:') as bdb:
            bayesdb_register_backend(bdb,
                LoomBackend(loom_store_path=loom_store_path))
            bdb.sql_execute('create table t (x)')
            for x in xrange(10):
                bdb.sql_execute('insert into t (x) values (?)', (x,))
            bdb.execute('create population p for t (x numerical)')
            bdb.execute('create generator g0 for p using loom')
            bdb.execute('create generator g1 for p using loom')
