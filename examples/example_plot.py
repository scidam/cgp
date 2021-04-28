#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Created Date: Wednesday March 31st 2021
Author: Dmitry Kislov
E-mail: kislov@easydan.com
-----
Last Modified: Thursday, April 1st 2021, 2:05:39 pm
Modified By: Dmitry Kislov
-----
Copyright (c) 2021
"""


from cgp.model import (GridMixin, Config, FunctionNode, InputNode, OutputNode,
                       Phenotype)
from cgp.utils import CGPPlotter

cfg = Config(pool_shape=(2, 3), input_size=3, output_size=2)
inode1 = InputNode(inputs=[1])
inode2 = InputNode(inputs=[2])
inode3 = InputNode(inputs=[3])

fnode11 = FunctionNode(inputs=[inode1, inode2, inode3],
                       function=lambda x1, x2, x3: x1 * x2 * x3)
fnode21 = FunctionNode(inputs=[inode1, inode3], function=lambda x1, x2: x1 + x2)

fnode12 = FunctionNode(inputs=[fnode11], function=lambda x1: x1)
fnode22 = FunctionNode(inputs=[fnode21, fnode11], function=lambda x1, x2: x1 + x2)

fnode13 = FunctionNode(inputs=[fnode22, fnode12], function=lambda x1, x2: x1 - x2)
fnode23 = FunctionNode(inputs=[fnode12], function=lambda x1: x1)

onode1 = OutputNode(inputs=[fnode13])
onode2 = OutputNode(inputs=[fnode22])


def _build_phenotype() -> Phenotype:
    phenotype = Phenotype(inputs=[inode1, inode2, inode3],
                pool=[fnode11, fnode21, fnode12,
                      fnode22, fnode13, fnode23],
                outputs=[onode1, onode2], config=cfg)
    return phenotype


phenotype = _build_phenotype()
plotter = CGPPlotter()
plotter.grid_plot(phenotype)

