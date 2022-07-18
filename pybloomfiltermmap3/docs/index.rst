.. pybloomfilter documentation master file, created by
   sphinx-quickstart on Tue Nov 26 10:03:21 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pybloomfiltermmap3: a fast implementation of Bloom filter for Python
====================================================================

`pybloomfiltermmap3` is a Python 3  fork of `pybloomfiltermmap` by `Michael Axiak (@axiak) <https://github.com/axiak>`_.

Bloom filter is a probabilistic data structure used to test whether an element is a member of a set.
The `wikipedia page <http://en.wikipedia.org/wiki/Bloom_filter>`_ has further information on their nature.

This module implements a Bloom filter in Python that's fast
and uses mmap files for better scalability.

Here's a quick example:

.. code-block:: python

    >>> from pybloomfilter import BloomFilter

    >>> bf = BloomFilter(10000000, 0.01, 'filter.bloom')
    >>> with open("/usr/share/dict/words") as f:
    >>>     for word in f:
    >>>         bf.add(word.rstrip())

    >>> print 'apple' in bf
    True

That wasn't so hard, was it? Now, there are a lot of other things we can do.
For instance, let's say we want to create a similar filter with just a few pieces of fruit:

.. code-block:: python

    >>> fruitbf = bf.copy_template("fruit.bloom")
    >>> fruitbf.update(("apple", "banana", "orange", "pear"))

    >>> print(fruitbf.to_base64())
    "eJzt2k13ojAUBuA9f8WFyofF5TWChlTHaPzqrlqFCtj6gQi/frqZM2N7aq3Gis59d2ye85KTRbhk"
    "0lyu1NRmsQrgRda0I+wZCfXIaxuWv+jqDxA8vdaf21HIOSn1u6LRE0VL9Z/qghfbBmxZoHsqM3k8"
    "N5XyPAxH2p22TJJoqwU9Q0y0dNDYrOHBIa3BwuznapG+KZZq69JUG0zu1tqI5weJKdpGq7PNJ6tB"
    "GKmzcGWWy8o0FeNNYNZAQpSdJwajt7eRhJ2YM2NOkTnSsBOCGGKIIYbY2TA663GgWWyWfUwn3oIc"
    "fyLYxeQwiF07RqBg9NgHrG5ba3jba5yl4zS2LtEMMcQQQwwxmRiBhPGOJOywIPafYhUwqnTvZOfY"
    "Zu40HH/YxDexZojJwsx6ObDcT7D8vVOtJBxiAhD/AjMmjeF2Wnqd+5RrHdo4azPEzoANabiUhh0b"
    "xBBDDDHEENsf8twlrizswEjDhnTbzWazbGKpQ5k07E9Ox2iFvXBZ2D9B7DawyqLFu5lshhhiiGUK"
    "a4nUloa9yxkwR7XhgPPXYdhRIa77uDtnyvqaIXalGK02ufv3J36GmsnG4lquPnN9gJo1VNxqgYbt"
    "ji/EC8s1PWG5fuVizW4Jox6/3o9XxBBDDLFbwcg9v/AwjrPHtTRsX34O01mxLw37bhCTjJk0+PLK"
    "08HYd4MYYojdKmYnBfjsktEpySY2tGGZzWaIIfYDGB271Yaieaat/AaOkNKb"


Why pybloomfilter?
------------------

As already mentioned, there are a couple reasons to use this module:

* It natively uses `mmaped files <http://en.wikipedia.org/wiki/Mmap>`_.
* It natively does the set things you want a Bloom filter to do.
* It is fast (see `benchmarks <http://axiak.github.io/pybloomfiltermmap/#benchmarks>`_).


Install
-------

Please note that this version is for Python 3.5 and over.
In case you are using Python 2, please see `pybloomfiltermmap <https://github.com/axiak/pybloomfiltermmap>`_.

To build and install::

    $ pip install pybloomfiltermmap3


Develop
-------

To develop you will need Cython. The setup.py script should automatically
build from Cython source if the Cython module is available.


.. toctree::
    :caption: Class Reference
    :maxdepth: 2

    ref

.. toctree::
    :hidden:
    :caption: Distribution
    :maxdepth: 1

    License <license>
    Authors <authors>
    Changelog <changelog>


.. _Python: http://docs.python.org/
