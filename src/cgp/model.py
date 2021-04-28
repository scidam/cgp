#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Created Date: Sunday March 28th 2021
Author: Dmitry Kislov
E-mail: kislov@easydan.com
-----
Last Modified: Wednesday, April 28th 2021, 10:27:12 am
Modified By: Dmitry Kislov
-----
Copyright (c) 2021
"""


from __future__ import annotations
from .bases import (AbstractNode, AbstractPhenotype, AbstractConfig,
                    AbstractGrid)
from dataclasses import dataclass, field
from typing import Any, List, Union, Tuple, Iterable, Optional
import operator
import logging
import random
from copy import deepcopy
import inspect
import re
import sys

__author__ = "Dmitry E. Kislov"
__copyright__ = "Dmitry E. Kislov"
__license__ = "MIT"

_logger = logging.getLogger(__name__)


def _get_function_arity(f) -> int:
    if hasattr(f, 'arity'):
        return f.arity
    for par in inspect.signature(f).parameters.values():
        if par.kind not in [
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]:
            raise TypeError("Node function should have only positional arguments.")
        elif par.kind == inspect.Parameter.VAR_POSITIONAL:
            return -1
    return len(inspect.signature(f).parameters)


def _assign_function_arities(function_table) -> None:
    for f in function_table:
        if not hasattr(f, 'arity'):
            setattr(f, 'arity', _get_function_arity(f))


@dataclass(eq=False, repr=False)    # type: ignore
class IONode(AbstractNode):
    """Input and output node class for CGP algorithm
    """

    length_constraint_err =\
        "Length of input or output nodes should be equal 1."

    def __post_init__(self) -> None:
        if len(self.inputs) != 1:
            raise ValueError(self.length_constraint_err)
        self.active = True

    def __setattr__(self, name, value):
        if name == 'inputs':
            if hasattr(value, '__len__'):
                if len(value) != 1:
                    raise ValueError(self.length_constraint_err)
            else:
                raise TypeError("Inputs expected to be Iterable"
                                " and have __len__ method.")
        super().__setattr__(name, value)

    @property
    def short_repr(self):
        return f"{self.inputs[0]}"


class InputNode(IONode):

    def evaluate(self) -> Any:
        self.output = self.inputs[0]
        self.evaluated = True
        return self.output


class OutputNode(IONode):

    def __setattr__(self, name, value):
        if name == 'inputs':
            if isinstance(value, Iterable):
                for item in value:
                    if not isinstance(item, (FunctionNode, InputNode)):
                        raise TypeError("Inputs in output nodes should"
                                        " be instances of FunctionNode"
                                        " or InputNode")
        super().__setattr__(name, value)

    def evaluate(self) -> Any:
        self.output = self.inputs[0].evaluate()
        self.evaluated = True
        return self.output


class FunctionNode(AbstractNode):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not hasattr(self, 'function'):
            raise ValueError("Functional node wasn't properly defined.")
        if not hasattr(self.function, 'arity'):
            _logger.info(f"{self.short_repr}:"
                         " the arity attribute was assigned.")
            setattr(self.function, 'arity', _get_function_arity(self.function))

    @property
    def short_repr(self):
        if hasattr(self.function, 'short_repr'):
            return self.function.short_repr
        else:
            # inspecting source code and  trying to
            # build its short representation
            src, _ = inspect.getsourcelines(self.function)
            src = src[-1].strip().replace('return ', '')
            match = re.findall(r".*lambda.*\:(.*)\s?\)?\s?$", src)
            if match:
                result = match[-1].strip()
            else:
                result = ""
            return result

    def evaluate(self) -> Any:
        if self.evaluated:
            return self.output
        args = []
        for node in self.inputs:
            if node.evaluated:
                args.append(node.output)
            else:
                args.append(node.evaluate())
        self.output = self.function(*args)
        self.evaluated = True
        return self.output

    @property
    def input_size(self):
        return len(self.inputs)


@dataclass
class Config(AbstractConfig):

    loglevel = logging.ERROR
    logto = sys.stdout

    def __log_config(self):
        try:
            _logger.setLevel(self.loglevel)
        except ValueError:
            _logger.setLevel(logging.INFO)
        if isinstance(self.logto, str):
            handler = logging.FileHandler(self.logto)
        else:
            handler = logging.StreamHandler(sys.stdout)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        _logger.addHandler(handler)

    def __post_init__(self):
        self.__log_config()
        if self.level_back == 0:
            self.level_back = self.pool_shape[1] - 1
        if self.conn_mutation_rate >= 1.0 or self.conn_mutation_rate < 0.0:
            raise\
                ValueError("Connection mutation rate doesn't belong to [0, 1).")
        if self.func_mutation_rate >= 1.0 or self.func_mutation_rate < 0.0:
            raise\
                ValueError("Function mutation rate doesn't belong to [0, 1).")
        if self.output_mutation_rate >= 1.0 or\
                self.output_mutation_rate < 0.0:
            raise ValueError("Output mutation rate doesn't belong to [0, 1).")
        if self.level_back < 0:
            raise IndexError("Level-back attribute should be non-negative 0. "
                             f"level_back = {self.level_back}.")
        _assign_function_arities(self.function_table)


    @property
    def size(self) -> int:
        return self.pool_shape[0] * self.pool_shape[1]


@dataclass
class GridMixin(AbstractGrid):
    pool: List[Any] = field(default_factory=list)
    config: Config = field(default_factory=Config)

    def __getitem__(self, ix: Union[int, Tuple[int, int]]) -> FunctionNode:
        if isinstance(ix, int):
            if ix < 0:
                raise IndexError("Indecies must be positive.")
            return self.pool[ix]
        elif isinstance(ix, tuple):
            if ix[1] < 0 or ix[1] < 0:
                raise IndexError("Indecies must be positive.")
            elif ix[0] >= self.config.pool_shape[0] or\
                    ix[1] >= self.config.pool_shape[1]:
                raise IndexError("Indecies out of range.")
            return self.pool[self.config.pool_shape[0] * ix[1] + ix[0]]
        else:
            raise IndexError("Only int and [int, int]"
                             " are allowed as indecies.")

    def get_node_column(self, n: Union[int, FunctionNode]) -> int:
        if isinstance(n, FunctionNode):
            ix = self.pool.index(n)
        elif isinstance(n, int):
            ix = n
        else:
            raise\
                TypeError(f"Illegal type of the function argument: {type(n)}.")
        if ix < 0 or ix >= (self.config.pool_shape[0] *
                            self.config.pool_shape[1]):
            raise IndexError(f"Index ({ix}) is out of bounds "
                             f"for shape {self.config.pool_shape}.")
        return ix // self.config.pool_shape[0]

    def get_node_row(self, n: Union[int, FunctionNode]) -> int:
        if isinstance(n, FunctionNode):
            ix = self.pool.index(n)
        elif isinstance(n, int):
            ix = n
        else:
            raise\
                TypeError(f"Illegal type of the function argument: {type(n)}.")
        if ix < 0 or ix >= (self.config.pool_shape[0] *
                            self.config.pool_shape[1]):
            raise IndexError(f"Index ({ix}) is out of bounds "
                             f"for shape {self.config.pool_shape}.")
        return ix % self.config.pool_shape[0]

    def get_levelback_nodes(
        self, n: Union[int, FunctionNode], level_back: int = 1
    ) -> List[FunctionNode]:
        col = self.get_node_column(n)
        return self.pool[max(col - level_back, 0) * self.config.pool_shape[0]:
                         col * self.config.pool_shape[0]]


class Phenotype(AbstractPhenotype, GridMixin):

    @property
    def active_func_nodes(self) -> List[FunctionNode]:
        if not all(map(operator.attrgetter('visited'), self.outputs)):
            self.refresh_node_states()
        return list(filter(operator.attrgetter('active'), self.pool))

    def output_mutate(self) -> None:
        to_mutate = int(
            self.config.output_size * self.config.output_mutation_rate
        )
        if to_mutate > 0:
            _logger.info(f"Phenotype (id={id(self)}): output mutate started."
                         f" {to_mutate} nodes will be changed.")
            allowed_nodes = [n for n in self.pool[-self.config.level_back *
                                                  self.config.pool_shape[0]:]
                             if n in self.active_func_nodes]
            allowed_nodes += self.inputs
            for output in random.sample(self.outputs, to_mutate):
                output.inputs = [random.choice(allowed_nodes)]
                output.evaluated = False
            self.refresh_node_states()
        else:
            _logger.info(f"Phenotype (id={id(self)}): output mutate:"
                         " nothing to mutate.")

    def func_mutate(self) -> None:
        to_mutate = int(len(self.active_func_nodes) *
                        self.config.func_mutation_rate)
        if to_mutate > 0:
            _logger.info(f"Phenotype (id={id(self)}): function mutate started."
                         f" {to_mutate} nodes will be changed.")
            for item in random.sample(self.active_func_nodes, to_mutate):
                arity = item.function.arity  # type: ignore
                funcs = list(filter(
                    lambda x: x.arity == arity or arity == -1 or x.arity == -1,
                    self.config.function_table
                ))
                if funcs:
                    setattr(item, 'function', random.choice(funcs))
                    self.reset_eval_statuses(item)
        else:
            _logger.info(f"Phenotype (id={id(self)}): function mutate:"
                         " nothing to mutate.")

    def pool_mutate(self) -> None:
        total_conns = sum(map(operator.attrgetter('input_size'),
                              self.pool))
        to_mutate = int(total_conns * self.config.conn_mutation_rate)
        n_mut = 1
        lback = self.config.level_back
        nrows, ncols = self.config.pool_shape[0], self.config.pool_shape[1]
        if to_mutate > 0:
            _logger.info(f"Phenotype (id={id(self)}): pool mutate started."
                         f" {to_mutate} nodes will be changed.")
            while n_mut <= to_mutate:
                #  --------- exchange mutation between active nodes ---------
                done = False
                cnt = 0
                while not done and cnt < 100:
                    cnt += 1
                    if not self.active_func_nodes:
                        break
                    node1 = random.choice(self.active_func_nodes)
                    node2 = random.choice(self.active_func_nodes)
                    if node1 is node2:
                        continue

                    # node1 and node2 candidate for connection exchange
                    inputs1 = node1.inputs
                    inputs2 = node2.inputs
                    filtered_i1 = []
                    for i1 in inputs1:
                        if isinstance(i1, InputNode):
                            filtered_i1.append(i1)
                        elif self.get_node_column(node2) - lback <\
                                self.get_node_column(i1) and\
                                self.get_node_column(i1) <\
                                self.get_node_column(node2):
                            filtered_i1.append(i1)

                    filtered_i2 = []
                    for i2 in inputs2:
                        if isinstance(i2, InputNode):
                            filtered_i2.append(i2)
                        elif self.get_node_column(node1) - lback <\
                                self.get_node_column(i2) and\
                                self.get_node_column(i2) <\
                                self.get_node_column(node1):
                            filtered_i2.append(i2)

                    if filtered_i1 and filtered_i2:
                        len_i1 = len(node1.inputs)
                        len_i2 = len(node2.inputs)
                        i1 = random.randint(0, len_i1 - 1)
                        i2 = random.randint(0, len_i2 - 1)
                        #plotter.grid_plot(self, filename=f"trav-{n_mut}-{cnt}-bra.pdf")
                        node1.inputs[i1] = random.choice(filtered_i2)
                        node2.inputs[i2] = random.choice(filtered_i1)
                        #plotter.grid_plot(self, filename=f"trav-{n_mut}-{cnt}-ra.pdf")
                        self.refresh_node_states()
                        n_mut += 2
                        done = True

                # ---------- change function arguments order ---------
                if self.active_func_nodes:
                    node1 = random.choice(self.active_func_nodes)
                    if node1.function.arity > 1 or node1.function.arity == -1:  # type: ignore   # noqa: E501
                        random.shuffle(node1.inputs)
                        if not done:
                            self.reset_eval_statuses(node1)
                        n_mut += 1
                        #plotter.grid_plot(self, filename=f"trav-{n_mut}-fa.pdf")
                # --------------- make node active -------------------
                na_nodes = list(filter(lambda x: x.active is False, self.pool))
                if na_nodes:
                    na_node = random.choice(na_nodes)
                    allowed = self.get_levelback_nodes(
                        na_node,
                        level_back=lback
                    ) + self.inputs
                    allowed = list(filter(operator.attrgetter('active'),
                                          allowed))
                    allowed += self.inputs
                    arity = na_node.function.arity
                    if arity == -1:
                        arity = random.randint(1, len(allowed))
                    na_node.inputs = random.sample(allowed, arity)
                    ix = min(self.get_node_column(na_node) + 1,
                             ncols - 1) * nrows
                    rest_part_nodes = list(filter(
                        operator.attrgetter('active'),
                        self.pool[ix:]
                    )) + self.outputs
                    rnode = random.choice(rest_part_nodes)
                    rnode.inputs[0] = na_node
                    if not done:
                        self.reset_eval_statuses(na_node)
                    n_mut += 1
                    if not self.is_traversable:
                        _logger.error("Phenotype became not traversable during mutation."
                                      "This is unexpected behavior...")
                        raise Exception("Something went wrong...")
            if done:
                self.reset_eval_statuses()
        else:
            _logger.info(f"Phenotype (id={id(self)}): pool mutate: "
                         "nothing to mutate.")

    def is_valid(self) -> bool:
        """Check if the phenotype instance is valid.

        It should be overridden according to the specific problem.

        :return: By default, it is shortcut for `is_traversable` method.
        :rtype: bool
        """
        return self.is_traversable

    def _require_traverse(self) -> None:
        """Mark nodes as <not visited>. Once <is_traversable> will be called,
        it traverses all the graph instead of returning cached results.
        """

        for node in self.outputs:
            node.visited = False
            node.active = False

        for node in self.pool:
            node.active = False
            node.visited = False

        for node in self.inputs:
            node.visited = True
            node.active = True

    @property
    def is_traversable(self) -> bool:

        def _traverse(node: Union[InputNode, FunctionNode, OutputNode]) -> bool:
            if not isinstance(node, InputNode) and not node.inputs:
                node.visited = True
                node.active = False
                return False
            elif isinstance(node, InputNode):
                node.visited = True
                node.active = True
                return True
            node.active = all(_traverse(n) for n in node.inputs)
            node.visited = True
            return node.active

        statuses = []
        for o in self.outputs:
            statuses.append(_traverse(o))
        return all(statuses)

    def refresh_node_states(self) -> None:
        self._require_traverse()
        self.is_traversable

    def update_inputs(self, inputs: List[Any]) -> None:
        for k in range(len(inputs)):
            if isinstance(inputs[k], Iterable):
                if any(self.inputs[k].inputs[0] != inputs[k]):
                    self.reset_eval_statuses()
                    self.inputs[k].inputs[0] = inputs[k]
                    self.inputs[k].evaluate()
            else:
                if self.inputs[k].inputs[0] != inputs[k]:
                    self.reset_eval_statuses()
                    self.inputs[k].inputs[0] = inputs[k]
                    self.inputs[k].evaluate()

    def reset_eval_statuses(
        self,
        from_node: Optional[Union[InputNode, FunctionNode, OutputNode]] = None
    ) -> None:
        """Sets `evaluted` flag to `False` for all nodes following
        the `from_node`.
        """

        reset_all_nodes = False

        if isinstance(from_node, InputNode):
            reset_all_nodes = True
        elif isinstance(from_node, OutputNode):
            return
        elif from_node is None:
            reset_all_nodes = True
        elif isinstance(from_node, FunctionNode):
            try:
                ncol = self.get_node_column(from_node)
            except ValueError:
                reset_all_nodes = True
            if not reset_all_nodes:
                from_node.evaluated = False
                for item in self.pool[self.config.pool_shape[0] * (ncol + 1):]:
                    item.evaluated = False

        if reset_all_nodes:
            for n in self.pool:
                n.evaluated = False

            for n in self.outputs:
                n.evaluated = False

    def evaluate(self, inputs: Optional[List[Any]] = None) -> List[Any]:
        """ Forward pass throgout the phenotype """

        if inputs is not None:
            self.update_inputs(inputs)

        return [node.inputs[0].evaluate() for node in self.outputs]

    def clone(self) -> Phenotype:
        return deepcopy(self)

    def random_init(self) -> None:

        # initialize input nodes
        self.inputs = []
        for k in range(self.config.input_size):
            inode = InputNode(inputs=[0])
            inode.acitve = True
            self.inputs.append(inode)

        self.pool = []
        for j in range(self.config.pool_shape[0] * self.config.pool_shape[1]):
            pnode = FunctionNode(
                inputs=[],
                function=random.choice(self.config.function_table)
            )
            self.pool.append(pnode)

        for node_index in range(
            0,
            self.config.pool_shape[0] * self.config.pool_shape[1],
            self.config.pool_shape[0]
        ):
            allowed_nodes = self.get_levelback_nodes(
                    node_index,
                    self.config.level_back
                ) + self.inputs

            allowed_nodes = list(filter(operator.attrgetter('active'),
                                        allowed_nodes))

            for k in range(self.config.pool_shape[0]):
                if random.random() < self.config.initialization_prob:
                    arity = _get_function_arity(
                        self.pool[k + node_index].function
                    )
                    if arity == -1:
                        arity = random.randint( # FIXME: ? hardcoded values ?
                            0,
                            min(len(allowed_nodes) // 2, 4) + 1
                        )
                    for _ in range(arity):
                        self.pool[
                            k + node_index
                        ].inputs.append(random.choice(allowed_nodes))
                    self.pool[k + node_index].active = True
                else:
                    self.pool[k + node_index].active = False

        self.outputs = []
        allowed_nodes = list(
            filter(
                operator.attrgetter('active'),
                self.pool[-self.config.level_back *
                          self.config.pool_shape[0]:] + self.inputs
            )
        )
        for _ in range(self.config.output_size):
            onode = OutputNode(inputs=[random.choice(allowed_nodes)])
            self.outputs.append(onode)

        self.refresh_node_states()


@dataclass
class Evolution:
    population: List[Phenotype] = field(default_factory=list)
    config: Config = field(default_factory=Config)
    init_population: bool = False
    history: set = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.init_population:
            self.initialize_population()
        if self.config.metric is None:
            raise ValueError("Metric is not defined (is None)."
                             "Define metric function and run again.")
        if self.config.select_criterion is None:
            self.config.select_criterion = lambda *x: False
            _logger.warning("The select_criterion function is not defined."
                         "Use always `False` function. Define select_criterion"
                         "and start the evolution again.")
        if self.config.stop_criterion is None:
            self.config.stop_criterion = lambda *x: True
            _logger.warning("The stop_criterion function is not defined."
                         "Use always `True` function. Define stop_criterion"
                         "and start the evolution again.")

    def get_random_phenotype(self) -> Phenotype:
        """Initialize random instance of Phenotype class
           according to the configs provided"""

        phenotype = Phenotype(config=self.config)
        phenotype.random_init()
        return phenotype

    def initialize_population(self) -> None:
        for cnt in range(self.config.max_population_size):
            self.population.append(self.get_random_phenotype())

    def crossover(
        self,
        g1: Phenotype,
        g2: Phenotype,
        cutoff: int = 0
    ) -> List[Phenotype]:

        if self.config.pool_shape[1] == 1:
            _logger.warning(
                "Couldn't perform crossover operation"
                f" over pool of shape ({self.config.pool_shape}).")
            return [g1, g2]

        if not cutoff:
            cutoff = random.randint(
                0,
                self.config.pool_shape[1] - 1
            ) * self.config.pool_shape[0]

        _g1 = g1.clone()
        _g2 = g2.clone()
        halfg1_1 = _g1.pool[:cutoff]
        halfg2_2 = _g2.pool[cutoff:]

        for node in halfg2_2 + _g2.outputs:
            for k in range(len(node.inputs)):
                if isinstance(node.inputs[k], FunctionNode):
                    index = _g2.pool.index(node.inputs[k])
                    if index < cutoff:
                        node.inputs[k] = _g1[index]
                elif isinstance(node.inputs[k], InputNode):
                    inp_ind = _g2.inputs.index(node.inputs[k])
                    node.inputs[k] = _g1.inputs[inp_ind]

        halfg1_2 = _g1.pool[cutoff:]
        halfg2_1 = _g2.pool[:cutoff]

        for node in halfg1_2 + _g1.outputs:
            for k in range(len(node.inputs)):
                if isinstance(node.inputs[k], FunctionNode):
                    index = _g1.pool.index(node.inputs[k])
                    if index < cutoff:
                        node.inputs[k] = _g2[index]
                elif isinstance(node.inputs[k], InputNode):
                    inp_ind = _g1.inputs.index(node.inputs[k])
                    node.inputs[k] = _g2.inputs[inp_ind]

        merged1 = g1.clone()
        merged1.inputs = _g1.inputs
        merged1.pool = halfg1_1 + halfg2_2
        merged1.outputs = _g2.outputs

        merged2 = g2.clone()
        merged2.inputs = _g2.inputs
        merged2.pool = halfg2_1 + halfg1_2
        merged2.outputs = _g1.outputs

        # Renew node statuses
        merged1.refresh_node_states()
        merged2.refresh_node_states()

        return list(filter(operator.attrgetter('is_traversable'),
                           [merged1, merged2]))

    def select(self) -> List[Phenotype]:
        selected = []
        for phenotype in self.population:
            try:
                select = self.config.select_criterion(phenotype, self.population)
            except Exception as exc:
                _logger.error(
                    f"Exception raised when executing `select` method: {exc}"
                )
                select = False
            if select:
                selected.append(phenotype)
        return selected

    def stop_condition(self) -> bool:
        for phenotype in self.population:
            try:
                stop = self.config.stop_criterion(phenotype)
            except Exception as exc:
                _logger.error(
                    f"Exception raised when cheking `stop_condition`: {exc}"
                )
                stop = False
            if stop:
                return True
        return False

    def extend_if_novel(self, ext: List[Phenotype]) -> int:
        cnt = 0
        for p in ext:
            if self.append_if_novel(p):
                cnt += 1
        return cnt

    def append_if_novel(self, phenotype: Phenotype) -> bool:
        novelty = self.config.novelty_function(phenotype)
        if novelty not in self.history:
            self.population.append(phenotype)
        else:
            return False
        return True

    def append_to_history(self, phenotypes: List[Phenotype]) -> None:
        for p in phenotypes:
            self.history.add(self.config.novelty_function(p))

    def rectify_population(self) -> None:
        metrics = []
        selected = []
        for p in self.population:
            try:
                metrics.append(self.config.metric(p))
                selected.append(p)
            except Exception as exc:
                _logger.error(f"Exception raised: {exc}")
        rectified = []
        # print("Metrics to be rectivied: ", Counter(metrics))
        # print(f"Len before rectification {len(selected)}")
        for val in set(metrics):
            candidates = [p for p, v in zip(selected, metrics) if v == val]
            if len(candidates) > 2:
                rectified.append(random.choice(candidates))
            else:
                rectified += candidates
        # print(f"Len after rectification {len(rectified)}")
        # ----- add random phenotypes which can be treated as <novel>
        self.population = rectified
        cnt = 0
        while len(self.population) < self.config.max_population_size:
            cnt += 1
            self.append_if_novel(self.get_random_phenotype())
            if cnt % 1000 == 0:
                _logger.info(
                    f"Searching for novel phenotypes. Iterations passed: {cnt}."
                )
        # print(f"Refill population: {len(self.population)}")

    def run(self, nmax: int = 100) -> Tuple[Phenotype, Any]:
        n = 0
        nmax = min(nmax, self.config.max_iterations)
        while n < nmax and not self.stop_condition():
            # print(f"Population size: {len(self.population)}")
            # print(f"{Counter(map(metric, self.population))}")
            selected = self.select()
            s = len(selected)
            # print(f"Selected only; {s}")
            new_children = []
            while s < self.config.max_population_size:
                # preform crossover
                if random.random() < self.config.crossover_prob\
                        and len(selected) >= 2:
                    a, b = random.sample(selected, 2)
                    crossover = self.crossover(a, b)
                    s += len(crossover)
                    new_children += crossover
                    continue

                # perform random mutation
                g = random.choice(self.population)
                child = g.clone()
                child.mutate()
                new_children.append(child)
                s += 1
            p, best = self.get_best_solution()
            # print(f"Selected: {len(selected)} ")
            history = [p for p in self.population
                       if p not in selected and self.config.metric(p) > best]
            self.append_to_history(history)
            self.population = selected[:]
            # num = self.extend_if_novel(new_children)
            # print(f"New phenotypes added: {num}")
            self.rectify_population()

            # print(f"""{n}: Population size: {len(self.population)}.
            #       History size: {len(self.history)};
            #       best: {self.get_best_solution(metric)[1]}""")
            n += 1
        return self.get_best_solution()

    def get_best_solution(self) -> Tuple[Phenotype, float]:

        results = []
        for phenotype in self.population:
            try:
                results.append((phenotype, self.config.metric(phenotype)))
            except (ZeroDivisionError,):
                pass
        return min(results, key=operator.itemgetter(1))
