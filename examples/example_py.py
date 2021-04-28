#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Created Date: Thursday April 1st 2021
Author: Dmitry Kislov
E-mail: kislov@easydan.com
-----
Last Modified: Friday, April 16th 2021, 11:04:11 am
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


all_x = [1, 2.0]


def metric(
    phenotype: Phenotype,
    inputs: Optional[List[Any]] = all_x,
) -> bool:
    return abs(phenotype.evaluate(inputs)[0] - np.pi)


def select_criterion(
    phenotype: Phenotype,
    population: List[Phenotype],
    inputs: Optional[List[Any]] = all_x,
) -> bool:
    result = []
    for p in population:
        result.append(metric(p))
    # print(result)
    return metric(phenotype) <= np.percentile(result, 25)

def stop_criterion(
    phenotype: Phenotype,
    inputs: Optional[List[Any]] = all_x,
    eps=1.0e-10
) -> bool:
    return metric(phenotype) < eps

plotter = CGPPlotter()
function_table = (
    lambda x1, x2: x1 + x2,
    # lambda x1, x2: x1 - x2,
    # lambda x1, x2: x1 * x2,
    # lambda x1, x2: x1 / x2 if x2 != 0 else 1.0,
    lambda x1: np.sqrt(abs(x1))
)

cfg = Config(
    pool_shape=(2, 20),
    input_size=2,
    output_size=1,
    conn_mutation_rate=0.5,
    func_mutation_rate=0.5,
    output_mutation_rate=0.5,
    function_table=function_table,
    crossover_prob=0.5,
    max_iterations=2000,
    max_population_size=50,
    novelty_function=metric,
    select_criterion=select_criterion,
    stop_criterion=stop_criterion,
    metric=metric
)


ev = Evolution(
    config=cfg,
    init_population=True,
)

print("prepared...")
p, val = ev.run(nmax=500)
plotter = CGPPlotter()
plotter.grid_plot(p, filename="best_pi.pdf", active_only=True)
print("Best value is: ", val)