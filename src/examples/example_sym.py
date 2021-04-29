#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Created Date: Thursday April 1st 2021
Author: Dmitry Kislov
E-mail: kislov@easydan.com
-----
Last Modified: Thursday, April 29th 2021, 9:38:34 am
Modified By: Dmitry Kislov
-----
Copyright (c) 2021
"""


from cgp.model import Config, Phenotype, Evolution
from cgp.utils import CGPPlotter
from typing import Optional, List, Any
import numpy as np


# ------------- Symbolic regression problem ----------------
@np.vectorize
def unknown_function(x):
    return x * (x - 1) - 1


accuracy = 1.0e-5
num_points = 20
input_x = [np.array([0.1 * (i - num_points/2) for i in range(num_points)])]
output_y = unknown_function(input_x[0])

# ----------------------------------------------------------


def metric(
    phenotype: Phenotype,
    inputs: Optional[List[Any]] = input_x,
) -> bool:
    return np.linalg.norm(phenotype.evaluate(inputs) - output_y)


# Evolution will stop, when norm of residual become less eps 1.0e-5.
def stop_criterion(
    phenotype: Phenotype,
    inputs: Optional[List[Any]] = input_x,
    eps=accuracy
) -> bool:
    return metric(phenotype, inputs) < eps


def select_criterion(
    phenotype: Phenotype,
    population: List[Phenotype],
    inputs: Optional[List[Any]] = input_x,
) -> bool:
    result = []
    for p in population:
        result.append(metric(p))
    return metric(phenotype) < np.percentile(result, 25)


# ------------------- CGP's configuration --------------
def div_func(x1, x2):
    return x1 / x2 if all(x2 != 0) else np.array([1.0] * len(x1))


setattr(div_func, 'short_repr', 'x1 / x2')

function_table = (
    lambda x1, x2: x1 + x2,
    lambda x1, x2: x1 - x2,
    lambda x1, x2: x1 * x2,
    div_func
)

cfg = Config(
    pool_shape=(1, 100),
    input_size=1,
    output_size=1,
    conn_mutation_rate=0.5,
    func_mutation_rate=0.5,
    output_mutation_rate=0.5,
    crossover_prob=0.5,
    function_table=function_table,
    max_iterations=2000,
    max_population_size=100,
    novelty_function=metric,
    # Note: novelty_function is used to memorize such cgp-graphs
    # which were already filtered during selection process. It is important
    # for including novel random phenotypes into the population.
    select_criterion=select_criterion,
    stop_criterion=stop_criterion,
    metric=metric
)

ev = Evolution(
    config=cfg,
    init_population=True,
)

print("Computations started ...")
p, val = ev.run()
plotter = CGPPlotter()
plotter.grid_plot(p, filename="best_symbolic.pdf", active_only=True)
print(f"Computations completed: accuracy = {val}")
