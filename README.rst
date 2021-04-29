===
cgp
===


Simple implementation of Cartesian Genetic Programming approach.
The core CGP classes doesn't depend on any third party libraries
and uses pure Python loops for evolution (it can be slow, when solving
most of applied problems).


Installation
============

.. code-block:: bash

    git clone https://github.com/scidam/cgp.git


Testing
=======

`pytest` and `pytest-cov` should be installed to run tests.


.. code-block:: bash

    cd cgp
    pytest



Run examples
============

Run `pi` approximation example:

.. code-block:: bash

    cd src
    python -m examples.example_pi


Run symbolic regression example:

.. code-block:: bash

    cd src
    python -m examples.example_sym



Author
------

    Dmitry Kislov <kislov@easydan.com>
