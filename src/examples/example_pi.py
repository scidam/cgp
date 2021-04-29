#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Created Date: Thursday April 1st 2021
Author: Dmitry Kislov
E-mail: kislov@easydan.com
-----
Last Modified: Thursday, April 29th 2021, 7:33:04 am
Modified By: Dmitry Kislov
-----
Copyright (c) 2021
"""


from cgp.model import Config, Phenotype, Evolution
from cgp.utils import CGPPlotter
from typing import Optional, List, Any
import numpy as np


# -------------------- Pi approximation problem -------------

# Problem statement:
# Approximate pi value using `1`, `2` as allowed numbers
# and `sqrt` and `+` operations.
# Desired approximation precision is `precision`.

inputs = [1, 2.0]
precision = 1.0e-10

function_table = (
    lambda x1, x2: x1 + x2,
    lambda x1: np.sqrt(abs(x1))
)
# ------------------------------------------------------------


def metric(
    phenotype: Phenotype,
    inputs: Optional[List[Any]] = inputs,
) -> bool:
    return abs(phenotype.evaluate(inputs)[0] - np.pi)


def select_criterion(
    phenotype: Phenotype,
    population: List[Phenotype],
    inputs: Optional[List[Any]] = inputs,
) -> bool:
    result = []
    for p in population:
        result.append(metric(p))

    # Lets select only those phenotypes that fall into first quartile
    # in terms of approximation metric
    return metric(phenotype) <= np.percentile(result, 25)


def stop_criterion(
    phenotype: Phenotype,
    inputs: Optional[List[Any]] = inputs,
    eps=precision
) -> bool:
    return metric(phenotype) < eps


# plotter instance (used to generate output-pdf file)
plotter = CGPPlotter()


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


print("Computations started ...")
p, val = ev.run(nmax=500)
plotter = CGPPlotter()
plotter.grid_plot(
    p,  # best graph
    filename="./best_pi.pdf",  # plot file
    active_only=True  # include only active nodes to plot
    )
print(f"Computations completed: accuracy = {val}")
