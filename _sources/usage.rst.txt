Usage
=====

Installation
------------

To use GNNFREE, first install it using pip:

.. code-block:: console

    $ pip install gnnfree

Creating recipes
----------------

To construct a dgl graph,
you can use the ``gnnfree.utils.construct_graph_from_edges()`` function:

.. py:function:: lumache.utils.construct_graph_from_edges(head, tail)

   Return a DGL graph specified by edges.

   :param head: Edge head.
   :param tail: Edge tail.
   :return: DGLGraph.
   :rtype: DGLGraph