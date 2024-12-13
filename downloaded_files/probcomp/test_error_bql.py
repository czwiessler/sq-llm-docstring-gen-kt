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

import pytest

import bayeslite

import test_core


def test_droppop_with_generators():
    with test_core.t1() as (bdb, _population_id, _generator_id):
        distinctive_name = 'frobbledithorpequack'
        bdb.execute('create generator %s for p1 using cgpm' %
            (distinctive_name,))
        with pytest.raises(bayeslite.BQLError):
            try:
                bdb.execute('drop population p1')
            except bayeslite.BQLError as e:
                assert 'generators' in str(e)
                assert distinctive_name in str(e)
                raise
