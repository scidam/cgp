#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Created Date: Sunday March 28th 2021
Author: Dmitry Kislov
E-mail: kislov@easydan.com
-----
Last Modified: Wednesday, April 28th 2021, 10:39:22 am
Modified By: Dmitry Kislov
-----
Copyright (c) 2021
"""

from __future__ import annotations
from abc import abstractmethod, ABC, abstractproperty
from dataclasses import dataclass, field
from typing import (List, Any, Tuple, Callable, Union, TypeVar, Generic,
                    Optional)
from copy import deepcopy

AG_co = TypeVar('AG_co', covariant=True)


@dataclass(eq=False, repr=False)  # type: ignore  # Seems to be a bug https://github.com/python/mypy/issues/5374  # noqa: E501
class AbstractNode(ABC, Generic[AG_co]):
    inputs: List[Union[AG_co, Any]] = field(default_factory=list)
    output: Any = None

    # True if the node took part in computations
    active: bool = False

    # Used in functional nodes; Node function.
    function: Callable[..., Any] = lambda x: 0

    # True if the value at the node is already computed
    evaluated: bool = False

    # Internally used when finding active nodes (taking part in computations)
    visited: bool = False

    @abstractmethod
    def evaluate(self):
        ...

    @abstractproperty
    def short_repr(self):
        ...

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return hash(id(self))

    def __str__(self) -> str:
        return f"{id(self)}"

    def clone(self):
        return deepcopy(self)


class AbstractGrid(ABC):
    ...


@dataclass
class AbstractConfig(ABC):
    # The number of input nodes
    input_size: int = 1

    # Grid shape of functional nodes
    pool_shape: Tuple[int, int] = (1, 1)

    # The number of output nodes
    output_size: int = 1

    # level-back parameter in CGP approach
    level_back: int = 0  # if 0 then it is equal pool_shape[1] - 1

    # Table of functions used for initilizing functional nodes
    function_table: Tuple[Callable] = (lambda x: x, )

    # ---- Self explanatory evolution-related probabilities
    mutation_rate: float = 0.01
    output_mutation_rate: float = 0.01
    func_mutation_rate: float = 0.01
    conn_mutation_rate: float = 0.01
    crossover_prob: float = 0.1
    # -----------------------------------------------------

    # The number of evolutionary iterations
    max_iterations: int = 100

    # desired population size
    max_population_size: int = 50

    # Defines how many connection to previous layers will be
    # created during initilization of random phenotypes
    initialization_prob = 0.9

    # ----------------------------------------------------------
    # The following functions control cgp-evolution process;
    # It is highly recommended to looked at examples, e.g. symbolic regression
    # example, to get understanding of how it works.

    # used to generate novel random phenotypes; when phenotype excluded
    # from the population, the value computed by `novelty_function` is stored
    # in history service attribute; New randomly generated phenotypes aren't
    # included to the population, if its novelty value is already exists in
    # history attribute.
    novelty_function: Callable[['AbstractPhenotype', List[Any]],
                               float] = lambda *x: 1.0

    # It is expected to be computed for current phenotype, taking
    # into account all phenotypes in the population.
    # If True, the phenotype will be included into next population
    # during process of evolution.
    # By default it is always False (so no phenotypes will be included
    # into population); one should redefine this function to work with cgp.
    select_criterion: Callable[[
            'AbstractPhenotype',
            List['AbstractPhenotype'],
            List[Any]], bool
    ] = lambda *x: False

    # If True, evlution process stops.
    stop_criterion: Optional[Callable[['AbstractPhenotype', List[Any]],
                                      bool]] = None

    # The metric function used to estimate the quality of cgp-evolution.
    # When dealing with problems of function approximation, e.g. symbolic
    # regression, it is expected to return, e.g. float value; However,
    # in common case, it could return any type.
    metric: Optional[Callable[['AbstractPhenotype', List[Any]], Any]] = None


@dataclass  # type: ignore
class AbstractPhenotype(ABC, Generic[AG_co]):
    inputs: List[AG_co] = field(default_factory=list)
    outputs: List[AG_co] = field(default_factory=list)

    # population pool
    pool: List[AG_co] = field(default_factory=list)

    # configuration parameters
    config: Any = None

    @abstractproperty
    def active_func_nodes(self) -> List[AG_co]:
        ...

    @abstractmethod
    def output_mutate(self):
        ...

    @abstractmethod
    def pool_mutate(self) -> None:
        ...

    @abstractmethod
    def func_mutate(self) -> None:
        ...

    def mutate(self) -> None:
        """ Performs general mutation step.

        It just includes self-explanatory simple mutation steps.
        """
        self.func_mutate()
        self.pool_mutate()
        self.output_mutate()

    @abstractmethod
    def is_valid(self) -> bool:
        ...

    @abstractproperty
    def is_traversable(self) -> bool:
        ...

    @abstractmethod
    def refresh_node_states(self) -> None:
        ...

    @abstractmethod
    def update_inputs(self, inputs: List[Any]) -> None:
        ...

    @abstractmethod
    def reset_eval_statuses(
        self,
        from_node: Optional[AG_co] = None
    ) -> None:
        ...

    @abstractmethod
    def evaluate(self, inputs: List[Any]) -> Any:
        ...

    @abstractmethod
    def clone(self) -> AbstractPhenotype:
        ...

    @abstractmethod
    def random_init(self) -> None:
        ...
