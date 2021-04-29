#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Created Date: Sunday March 28th 2021
Author: Dmitry Kislov
E-mail: kislov@easydan.com
-----
Last Modified: Thursday, April 29th 2021, 7:13:18 am
Modified By: Dmitry Kislov
-----
Copyright (c) 2021
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional


try:
    import networkx as nx
    from networkx.drawing.nx_agraph import to_agraph
except ImportError:
    nx = None

_logger = logging.getLogger(__name__)


@dataclass
class CGPPlotter:
    filename: str = 'output.pdf'
    active_color: str = '#0000ff'
    passive_color: str = "#ff0000"
    prog: str = 'dot'

    def grid_plot(
        self,
        phenotype: Any,
        filename: Optional[str] = None,
        active_only: bool = False
    ) -> None:
        if nx is None:
            _logger.error("Networkx is not installed.")
            return
        if filename:
            _logger.info(f"Setting filename: {filename}.")
            self.filename = filename
        phenotype.refresh_node_states()
        graph = nx.DiGraph()
        for ind, n in enumerate(phenotype.inputs):
            _logger.info(f"Including input node, id: {id(n)}.")
            graph.add_node(n,
                           pos=f"{0},{ind}!",
                           shape="square",
                           label=f"in[{ind+1}]\n({n.inputs[0]})")

        for ind, n in enumerate(phenotype.pool):
            if active_only and not n.active:
                _logger.info(
                    f"Excluding non-active functional node, id: {id(n)}."
                )
                continue
            i, j = phenotype.get_node_row(ind), phenotype.get_node_column(ind)
            graph.add_node(
                n,
                stype="filled",
                shape="circle",
                label=f"[{i + 1}, {j + 1}]\n{n.short_repr}",
                fillcolor=self.active_color
                if n.active else self.passive_color,
                color=self.active_color if n.active else self.passive_color
            )
            _logger.info(f"Including functional node, id: {id(n)}.")

        for ind, n in enumerate(phenotype.outputs):
            graph.add_node(n, shape="square",
                           label=f"out[{ind}]")
            _logger.info(f"Including output node, id: {id(n)}.")

        # add edges
        for n2 in phenotype.pool + phenotype.outputs:
            for ind, n1 in enumerate(n2.inputs):
                if active_only and (not n1.active or not n2.active):
                    continue
                if isinstance(n2, type(phenotype.pool[0])):
                    graph.add_edge(n1, n2, label=f"x{ind+1}")
                else:
                    graph.add_edge(n1, n2)
                _logger.info(
                    "Inserting edge between"
                    f" node1({id(n1)}) and node2({id(n2)})."
                )

        out = to_agraph(graph)
        out.graph_attr.update(splines=True, overlap=False)
        out.layout(prog=self.prog)
        out.draw(self.filename, prog=self.prog, args="-Knop -Tpdf")
        _logger.info(f"Graph created: {self.filename}.")
        return graph
