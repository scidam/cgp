#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Created Date: Thursday April 1st 2021
Author: Dmitry Kislov
E-mail: kislov@easydan.com
-----
Last Modified: Thursday, April 15th 2021, 9:23:22 am
Modified By: Dmitry Kislov
-----
Copyright (c) 2021
"""



from cgp.model import (GridMixin, Config, FunctionNode, InputNode, OutputNode,
                       Phenotype, Evolution)
from cgp.utils import CGPPlotter
from math import pi
from typing import Optional, List, Any
import numpy as np
import time
from collections import Counter
import sys


# ------------- Symbolic regression problem ----------------
@np.vectorize
def unknown_function(x):
    return x * (x + 1) + 1


num_points = 20
all_x = [np.array([0.1 * (i - num_points/2) for i in range(num_points)])]
all_y = unknown_function(all_x[0])

# So, we need to reconstruct unknown_function by its values at 20 points...

# ----------------------------------------------------------


def metric(
    phenotype: Phenotype,
    inputs: Optional[List[Any]] = all_x,
) -> bool:
    return np.linalg.norm(phenotype.evaluate(inputs) - all_y)


# Evolution will stop, when norm of residual become less eps 1.0e-5.
def stop_criterion(
    phenotype: Phenotype,
    inputs: Optional[List[Any]] = all_x,
    eps=1.0e-5
) -> bool:
    return metric(phenotype, inputs) < eps



# Here we choose only 25% among all phenotypes to include to the next
# population
def select_criterion(
    phenotype: Phenotype,
    population: List[Phenotype],
    inputs: Optional[List[Any]] = all_x,
) -> bool:
    result = []
    for p in population:
        result.append(metric(p))
    return metric(phenotype) < np.percentile(result, 25)

# ------------------- CGP's configuration --------------


function_table = (
    lambda x1, x2: x1 + x2,
    lambda x1, x2: x1 - x2,
    lambda x1, x2: x1 * x2,
    lambda x1, x2: x1 / x2 if all(x2 != 0) else np.array([1.0] * len(x1))
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
    novelty_function=metric
    # Note: novelty_function is used to memorize such cgp-graphs
    # which were already filtered during selection process. It is important
    # for including novel random phenotypes into the population.
)

ev = Evolution(
    config=cfg,
    init_population=True,
    select_criterion=select_criterion,
    stop_criterion=stop_criterion,
    metric=metric
)

print("prepared...")
p, val = ev.run(metric)
plotter = CGPPlotter()
plotter.grid_plot(p, filename="best_symbolic.pdf", active_only=True)
print("Best value is: ", val)