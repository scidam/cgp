#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Created Date: Sunday March 28th 2021
Author: Dmitry Kislov
E-mail: kislov@easydan.com
-----
Last Modified: Wednesday, April 28th 2021, 8:30:31 am
Modified By: Dmitry Kislov
-----
Copyright (c) 2021
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from cgp.model import (GridMixin, Config, FunctionNode, InputNode, OutputNode,
                       Phenotype, Evolution)
from cgp.utils import CGPPlotter
import numpy as np
from copy import deepcopy
import random

random.seed(42)
plotter = CGPPlotter()

__author__ = "Dmitry E. Kislov"
__copyright__ = "Dmitry E. Kislov"
__license__ = "MIT"


# ---------------- GridMixin tests --------------------------
pool_shape = (10, 20)
mx = GridMixin(
    pool=np.arange(pool_shape[0] * pool_shape[1]),
    config=Config(pool_shape=pool_shape)
)


def test_get_node_row():
    assert mx.get_node_row(5) == 5
    assert mx.get_node_row(0) == 0
    assert mx.get_node_row(10) == 0
    assert mx.get_node_row(15) == 5
    assert mx.get_node_row(1) == 1
    assert mx.get_node_row(199) == 9

    with pytest.raises(IndexError, match=r".* out of bounds .*"):
        mx.get_node_row(200)


def test_get_node_column():

    assert mx.get_node_column(0) == 0
    assert mx.get_node_column(5) == 0
    assert mx.get_node_column(10) == 1
    assert mx.get_node_column(49) == 4
    assert mx.get_node_column(50) == 5

    with pytest.raises(IndexError, match=r".* out of bounds .*"):
        mx.get_node_column(-1)

    with pytest.raises(IndexError, match=r".* out of bounds .*"):
        mx.get_node_column(200)


def test_levelback_nodes():

    nodes = mx.get_levelback_nodes(30, level_back=2)
    expected = list(range(10, 30))
    assert np.allclose(nodes, expected)

    nodes_full = mx.get_levelback_nodes(pool_shape[0] * pool_shape[1] - 1,
                                        level_back=19)
    expected = list(range(0, pool_shape[0] * (pool_shape[1] - 1)))
    assert np.allclose(nodes_full, expected)

    with pytest.raises(IndexError):
        mx.get_levelback_nodes(400, level_back=2)

    with pytest.raises(IndexError):
        mx.get_levelback_nodes(-1, level_back=2)


# -----------------------------------------------------------

# -------------------- Config tests -------------------------


def test_config_mutation_rates():
    with pytest.raises(ValueError, match=r"Function.*"):
        Config(func_mutation_rate=1.1)
    with pytest.raises(ValueError, match=r"Connection.*"):
        Config(conn_mutation_rate=-0.1)
    with pytest.raises(ValueError, match=r"Output.*"):
        Config(output_mutation_rate=100)

    cfg = Config(output_mutation_rate=0.9,
                 func_mutation_rate=0.0, conn_mutation_rate=0)
    assert cfg.func_mutation_rate == 0.0
    assert cfg.conn_mutation_rate == 0


# -----------------------------------------------------------


# -------------------- Node tests ---------------------------
def test_input_node():
    inode = InputNode(inputs=[1])
    assert inode.evaluate() == 1

    inode.inputs = [1.5]
    assert inode.evaluate() == 1.5

    with pytest.raises(Exception, match=r".*should be equal 1.*"):
        InputNode(inputs=[object, object])

    with pytest.raises(Exception, match=r".*should be equal 1.*"):
        inode.inputs = [object, object]

    with pytest.raises(Exception, match=r".*__len__ method.*"):
        inode.inputs = 5

    with pytest.raises(Exception, match=r".*__len__ method.*"):
        inode.inputs = (k for k in range(5))


def test_func_node_simplest():
    inode = InputNode(inputs=[2])
    # test simplest case
    fnode1 = FunctionNode(inputs=[inode], function=lambda x: x * x)
    assert fnode1.evaluate() == 4
    assert fnode1.output == 4


def test_func_node_simple_chain():
    inode = InputNode(inputs=[2])
    # test simplest case
    fnode1 = FunctionNode(inputs=[inode], function=lambda x: x * x)
    fnode2 = FunctionNode(inputs=[fnode1], function=lambda x: x - 1)
    assert fnode2.evaluate() == 3


def test_func_node_complex_chain():
    inode1 = InputNode(inputs=[2])
    inode2 = InputNode(inputs=[3])

    fnode1 = FunctionNode(inputs=[inode1, inode2], function=lambda x, y: x * y)
    fnode2 = FunctionNode(inputs=[fnode1, inode1, inode2],
                          function=lambda x, y, z: x * y * z - 1)
    fnode3 = FunctionNode(inputs=[fnode1, inode1, inode2],
                          function=lambda x, y: x * y - 1)
    assert fnode2.evaluate() == 35
    with pytest.raises(TypeError):
        fnode3.evaluate()


def test_output_node():
    inode = InputNode(inputs=[1])
    onode = OutputNode(inputs=[inode])
    assert onode.evaluate() == 1


def test_output_node_chain():
    inode1 = InputNode(inputs=[2])
    inode2 = InputNode(inputs=[3])
    fnode1 = FunctionNode(inputs=[inode1, inode2], function=lambda x, y: x * y)
    fnode2 = FunctionNode(inputs=[fnode1, inode1, inode2],
                          function=lambda x, y, z: x * y * z - 1)
    onode1 = OutputNode(inputs=[fnode1])
    onode2 = OutputNode(inputs=[fnode2])

    assert onode1.evaluate() == 6
    assert onode2.evaluate() == 35

    with pytest.raises(Exception, match=r".*should be equal 1.*"):
        OutputNode(inputs=[fnode1, inode1])

    with pytest.raises(TypeError,
                       match=r".*be instances of FunctionNode or InputNode.*"):
        OutputNode(inputs=[2])


def test_input_node_clone():
    inode1 = InputNode(inputs=[2])
    inode2 = inode1.clone()
    inode1.inputs[0] = 3
    assert inode2.inputs[0] == 2


def test_func_node_clone():
    inode1 = InputNode(inputs=[2])
    inode2 = InputNode(inputs=[3])
    fnode1 = FunctionNode(inputs=[inode1, inode2])

    fnode2 = fnode1.clone()
    fnode1.inputs[0].inputs[0] = 10

    assert inode1.inputs[0] == 10
    assert fnode2.inputs[0].inputs[0] == 2
    assert fnode2.inputs[0] is not inode1


def test_output_node_clone():
    inode1 = InputNode(inputs=[2])
    inode2 = InputNode(inputs=[3])
    fnode1 = FunctionNode(inputs=[inode1, inode2])
    onode1 = OutputNode(inputs=[fnode1])  # noqa: F841
    onode2 = onode1.clone()
    onode1.inputs[0] = inode1
    assert onode2.inputs[0].inputs[0] is not inode1

# -----------------------------------------------------------


# -------------------- Phenotype tests ---------------------------
cfg = Config(
    pool_shape=(2, 3),
    input_size=3,
    output_size=2,
    function_table = (
        lambda x1, x2, x3: x1 * x2 * x3,
        lambda x1, x2: x1 + x2,
        lambda x1: x1,
        lambda x1, x2: x1 - x2
    )
)
inode1 = InputNode(inputs=[1])
inode2 = InputNode(inputs=[2])
inode3 = InputNode(inputs=[3])

fnode11 = FunctionNode(inputs=[inode1, inode2, inode3],
                       function=cfg.function_table[0])
fnode21 = FunctionNode(inputs=[inode1, inode3],
                       function=cfg.function_table[1])

fnode12 = FunctionNode(inputs=[fnode11], function=cfg.function_table[2])
fnode22 = FunctionNode(inputs=[fnode21, fnode11], function=cfg.function_table[1])

fnode13 = FunctionNode(inputs=[fnode22, fnode12], function=cfg.function_table[3])
fnode23 = FunctionNode(inputs=[fnode12], function=cfg.function_table[2])

onode1 = OutputNode(inputs=[fnode13])
onode2 = OutputNode(inputs=[fnode22])


def _build_phenotype() -> Phenotype:
    phenotype = Phenotype(
        inputs=[inode1, inode2, inode3],
        pool=[fnode11, fnode21, fnode12,
              fnode22, fnode13, fnode23],
        outputs=[onode1, onode2], config=cfg
    )
    return phenotype


default_phenotype = _build_phenotype().clone()
# ppl = CGPPlotter(filename="default.pdf")
# ppl.grid_plot(default_phenotype)
# ddffdfd

def build_non_traversable_phenotype() -> Phenotype:
    phenotype = default_phenotype.clone()
    phenotype[1, 0].inputs = []
    phenotype[1, 1].inputs = [phenotype[1, 0]]
    return phenotype


def test_simple_phenotype_state():
    phenotype = _build_phenotype()

    assert all([g.active for g in phenotype.outputs])
    assert all([g.active for g in phenotype.inputs])
    assert all([g.active for g in phenotype.pool]) is False

    assert phenotype[0, 0] is fnode11
    assert phenotype[1, 1] is fnode22
    assert phenotype[1, 2].active is False


def test_phenotype_indexing():
    phenotype = _build_phenotype()

    with pytest.raises(IndexError, match=r'.*Indecies must be positive.*'):
        phenotype[-1]

    with pytest.raises(IndexError, match=r'.*out of range.*'):
        phenotype[10]

    with pytest.raises(IndexError, match=r'Only int and \[int, int\] .*'):
        phenotype[...]

    with pytest.raises(IndexError, match=r'Only int and \[int, int\] .*'):
        phenotype[:]

    with pytest.raises(IndexError, match=r'.*Indecies must be positive.*'):
        phenotype[1, -1]

    with pytest.raises(IndexError, match=r'Indecies out of range.'):
        phenotype[2, 2]


def test_refresh_active_nodes():
    phenotype = default_phenotype.clone()
    phenotype.refresh_node_states()
    assert all(phenotype[k].active for k in range(5))
    assert phenotype[5].active is False


def test_phenotype_traversability_simple():
    phenotype = _build_phenotype()
    assert phenotype.is_traversable


def test_phenotype_non_traversable():
    phenotype = build_non_traversable_phenotype()
    phenotype._require_traverse()
    assert phenotype.is_traversable is False


def test_func_mutation():
    def func_sum_all(*args):
        return sum(args)
    func_sum_all.arity = -1
    phenotype = default_phenotype.clone()
    phenotype.config.function_table = (func_sum_all, )
    phenotype.config.func_mutation_rate = 0.5
    phenotype.func_mutate()
    assert any([n.function is func_sum_all for n in phenotype.pool])


def test_output_mutation():
    phenotype = _build_phenotype()
    pre_outputs = set(map(id, sum([n.inputs for n in phenotype.outputs], [])))
    phenotype.config.output_mutation_rate = 0.5
    phenotype.output_mutate()
    pre_outputs_new = set(map(id, sum([n.inputs for n in phenotype.outputs], [])))
    assert pre_outputs_new != pre_outputs


def test_pool_mutate():
    phenotype = default_phenotype.clone()
    phenotype.config.conn_mutation_rate = 0.5
    phenotype.config.level_back = 1
    old_connections = list(map(id, sum([n.inputs for n in phenotype.pool], [])))
    phenotype.pool_mutate()
    new_connections = list(map(id, sum([n.inputs for n in phenotype.pool], [])))
    assert old_connections != new_connections
    assert sum(x != y for x, y in zip(old_connections, new_connections)) >= 4


def test_common_mutate():
    phenotype = default_phenotype.clone()
    phenotype.config.conn_mutation_rate = 0.5
    phenotype.config.func_mutation_rate = 0.5
    phenotype.config.output_mutation_rate = 0.5
    old_conn_state = list(map(id, sum([k.inputs for k in phenotype.pool], [])))
    old_output_state = list(map(id, sum([k.inputs for k in phenotype.outputs], [])))
    print("Results: ", phenotype.is_traversable)
    phenotype.mutate()
    new_conn_state = list(map(id, sum([k.inputs for k in phenotype.pool], [])))
    new_output_state = list(map(id, sum([k.inputs for k in phenotype.outputs], [])))
    assert old_conn_state != new_conn_state
    assert old_output_state != new_output_state


def test_evaluate_entire_phenotype():

    result = default_phenotype.evaluate([0, 0, 0])
    assert result == [0, 0]

    result = default_phenotype.evaluate([1, 1, 1])
    assert result == [2, 3]

    result = default_phenotype.evaluate([10, 10, 10])
    assert result == [20, 1020]


def test_clone_phenotype():
    old_phenotype = default_phenotype
    new_phenotype = old_phenotype.clone()
    old_phenotype[0, 0].function = lambda x, y, z: 0
    old_phenotype[1, 0].function = lambda x, y: 0
    assert old_phenotype.evaluate([1, 1, 1]) == [0, 0]
    assert new_phenotype.evaluate([1, 1, 1]) != [0, 0]
    old_phenotype[0, 2].inputs = []
    old_phenotype._require_traverse()
    assert old_phenotype.is_traversable is False
    assert new_phenotype.is_traversable

# -----------------------------------------------------------


# ----------------- test evolutionary functions -------------
function_table = [
    lambda x1, x2: x1 + x2,
    lambda x1, x2: x1 - x2,
    lambda x1, x2: x1 * x2,
    lambda x1, x2: x1 / x2 if x2 != 0 else x1
]

function_table[0].short_repr = 'x1 + x2'
function_table[1].short_repr = 'x1 - x2'

function_table[2].short_repr = 'x1 * x2'
function_table[3].short_repr = 'x1 / x2'


def build_test_ev_object():
    cfg = Config(pool_shape=(5, 10), input_size=4, output_size=4,
                 max_population_size=10, function_table=function_table,
                 level_back=1, conn_mutation_rate=0.7, func_mutation_rate=0.7,
                 metric=lambda x: 1
                 )

    return Evolution(config=cfg, init_population=True)


# ---- Initial data for unit testing of the evolution class ---------
cfg2_2 = Config(pool_shape=(2, 2), input_size=2, output_size=2,
                max_population_size=4, function_table=function_table,
                level_back=1, conn_mutation_rate=0.5, func_mutation_rate=0.5,
                metric=lambda x: 1
                )

phenotype1 = Phenotype(config=cfg2_2)
inode1 = InputNode(inputs=[1])  # expected output [0, 2]
inode2 = InputNode(inputs=[1])
phenotype1.inputs = [inode1, inode2]
fnode1 = FunctionNode(function=function_table[0], inputs=[inode1, inode2])
fnode2 = FunctionNode(function=function_table[1], inputs=[inode1, inode2])
fnode3 = FunctionNode(function=function_table[2], inputs=[fnode1, fnode2])
fnode4 = FunctionNode(function=function_table[3], inputs=[fnode1, inode1])
phenotype1.pool = [fnode1, fnode2, fnode3, fnode4]
onode1 = OutputNode(inputs=[fnode3])
onode2 = OutputNode(inputs=[fnode4])
phenotype1.outputs = [onode1, onode2]

phenotype2 = Phenotype(config=cfg2_2)
inode1 = InputNode(inputs=[1])  # expected output [0, 0]
inode2 = InputNode(inputs=[1])
phenotype2.inputs = [inode1, inode2]
fnode1 = FunctionNode(function=function_table[0], inputs=[inode1, inode2])
fnode2 = FunctionNode(function=function_table[1], inputs=[inode1, inode2])
fnode3 = FunctionNode(function=function_table[2], inputs=[fnode1, fnode2])
fnode4 = FunctionNode(function=function_table[3], inputs=[fnode2, fnode1])
phenotype2.pool = [fnode1, fnode2, fnode3, fnode4]
onode1 = OutputNode(inputs=[fnode3])
onode2 = OutputNode(inputs=[fnode4])
phenotype2.outputs = [onode1, onode2]

phenotype3 = Phenotype(config=cfg2_2)
inode1 = InputNode(inputs=[1])  # expected output [1, 1]
inode2 = InputNode(inputs=[1])
phenotype3.inputs = [inode1, inode2]
fnode1 = FunctionNode(function=function_table[0], inputs=[inode1, inode2])
fnode2 = FunctionNode(function=function_table[1], inputs=[inode1, inode2])
fnode3 = FunctionNode(function=function_table[2], inputs=[inode1, inode2])
fnode4 = FunctionNode(function=function_table[3], inputs=[inode1, inode2])
phenotype3.pool = [fnode1, fnode2, fnode3, fnode4]
onode1 = OutputNode(inputs=[fnode3])
onode2 = OutputNode(inputs=[fnode4])
phenotype3.outputs = [onode1, onode2]

phenotype4 = Phenotype(config=cfg2_2)
inode1 = InputNode(inputs=[1])  # expected output [1, 1/2]
inode2 = InputNode(inputs=[1])
phenotype4.inputs = [inode1, inode2]
fnode1 = FunctionNode(function=function_table[0], inputs=[inode1, inode2])
fnode2 = FunctionNode(function=function_table[1], inputs=[inode1, inode2])
fnode3 = FunctionNode(function=function_table[2], inputs=[inode1, inode2])
fnode4 = FunctionNode(function=function_table[3], inputs=[inode1, fnode1])
phenotype4.pool = [fnode1, fnode2, fnode3, fnode4]
onode1 = OutputNode(inputs=[fnode3])
onode2 = OutputNode(inputs=[fnode4])
phenotype4.outputs = [onode1, onode2]


def test_build_init_population():
    ev = build_test_ev_object()
    assert all(isinstance(k, Phenotype) for k in ev.population)
    assert len(ev.population) == 10


def test_select_condition():
    ev = build_test_ev_object()

    # by defualt select function returns nothing (emtpy list)
    assert ev.select() == []

    # ------- Build specific population -----------
    ev_obj = Evolution(
        population=[
            phenotype1,
            phenotype2,
            phenotype3,
            phenotype4
        ],
        config=cfg2_2
    )

    # monkey patch select condition
    def select_criterion(phenotype, population,
                         phenotype_inputs=[], target_value=0, accuracy=2):
        value = phenotype.evaluate(inputs=phenotype_inputs)
        return abs(sum(value) - target_value) < accuracy

    ev_obj.config.select_criterion = select_criterion
    assert ev_obj.select() == [phenotype2, phenotype4]


def test_crossover_minimal():
    ev_obj = Evolution(
        population=[
            phenotype1,
            phenotype2,
        ],
        config=cfg2_2
    )
    a, b = ev_obj.crossover(*ev_obj.population, cutoff=0)
    assert a.evaluate() == ev_obj.population[1].evaluate()
    assert b.evaluate() == ev_obj.population[0].evaluate()
