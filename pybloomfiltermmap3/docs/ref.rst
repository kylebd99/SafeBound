===========================
BloomFilter Class Reference
===========================

.. module:: pybloomfilter
   :platform: Unix, Windows
   :synopsis: a fast implementation of Bloom filter for Python

.. class:: BloomFilter(capacity: int, error_rate: float, [filename = None: string], [perm=0755], [hash_seeds = None: list])

    Creates a new BloomFilter object with a given capacity and error_rate.

    :param int capacity: the maximum number of elements this filter
        can contain while keeping the false positive rate under ``error_rate``.
    :param float error_rate: false positive probability that will hold
        given that ``capacity`` is not exceeded. Must be between 0-1.
    :param str filename: filename to use to create the new Bloom filter.
        If a filename is not provided, an in-memory Bloom filter will be created.
    :param int perm: (*not applicable for an in-memory Bloom filter*)
        file access permission flags.
    :param list hash_seeds: optionally specify hash seeds to use for the
        hashing algorithm. Each hash seed must not exceed 32 bits. The number
        of hash seeds will determine the number of hashes performed.

    **Note that we do not check capacity.** This is important, because
    we want to be able to support logical OR and AND (see :meth:`BloomFilter.union`
    and :meth:`BloomFilter.intersection`). The capacity and error_rate then together
    serve as a contract -- you add less than capacity items, and the Bloom filter
    will have an error rate less than error_rate.

    Raises :class:`OSError` if the supplied filename does not exist or if user
    lacks permission to access such file. Raises :class:`ValueError` if the supplied
    ``error_rate`` is invalid, ``hash_seeds`` does not contain valid hash seeds, or
    if the file cannot be read.

Class Methods
-------------

.. classmethod:: BloomFilter.open(filename, [mode="rw"])

    Creates a :class:`BloomFilter` object from an existing file.

    :param str filename: existing filename
    :param str mode: file access mode
    :rtype: :class:`BloomFilter`

.. classmethod:: BloomFilter.from_base64(filename, string, [perm=0755])

    Unpacks the supplied base64 string (as returned by :meth:`BloomFilter.to_base64`)
    into the supplied filename and return a :class:`BloomFilter` object using that
    file.

    Example::

        >>> bf = BloomFilter.from_base64("/tmp/mike.bf",
            "eJwFwcuWgiAAANC9v+JCx7By0QKt0GHEbKSknflAQ9QmTyRfP/fW5E9XTRSX"
            "qcLlqGNXphAqcfVH\nRoNv0n4JlTpIvAP0e1+RyXX6I637ggA+VPZnTYR1A4"
            "Um5s9geYaZZLiT208JIiG3iwhf3Fwlzb3Y\n5NRL4uNQS6/d9OvTDJbnZMnR"
            "zcrplOX5kmsVIkQziM+vw4hCDQ3OkN9m3WVfPWzGfaTeRftMCLws\nPnzEzs"
            "gjAW60xZTBbj/bOAgYbK50PqjdzvgHZ6FHZw==\n")
        >>> "MIKE" in bf
        True

    :param str filename: new filename
    :param int perm: file access permission flags
    :rtype: :class:`BloomFilter`


Instance Attributes
-------------------

.. attribute:: BloomFilter.capacity -> int

    The maximum number of elements this filter can contain while keeping
    the false positive rate under :attr:`BloomFilter.error_rate`. Returns an integer.

.. attribute:: BloomFilter.error_rate -> float

    The acceptable probability of false positives. Returns a float.

.. attribute:: BloomFilter.bit_array -> int

    Bit vector representation of the Bloom filter contents. Returns an integer.

.. attribute:: BloomFilter.hash_seeds -> list

    Integer seeds used for the random hashing. Returns a list of integers.

.. attribute:: BloomFilter.filename -> string

    File name (compatible with file objects). Does not apply to an in-memory
    :class:`BloomFilter` and will raise :class:`ValueError` if accessed.
    Returns a string.

.. attribute:: BloomFilter.num_bits -> int

    Number of bits used in the filter as buckets. Returns an integer.

.. attribute:: BloomFilter.num_hashes -> int

    Number of hash functions used when computing. Returns an integer.

.. attribute:: BloomFilter.read_only -> bool

    Indicates if the opened :class:`BloomFilter` is read-only.
    Always ``False`` for an in-memory :class:`BloomFilter`.

.. attribute:: BloomFilter.name -> bytes

    PENDING DEPRECATION: use :meth:`BloomFilter.filename` instead.

    File name (compatible with file objects). Does not apply to an in-memory
    :class:`BloomFilter` and will raise :class:`ValueError` if accessed.
    Returns an encoded string.


Instance Methods
----------------

.. method:: BloomFilter.add(item)

    Adds an item to the Bloom filter.
    Returns a boolean indicating whether this item was present
    in the Bloom filter prior to adding (see :meth:`BloomFilter.__contains__`).

    :param item: hashable object
    :rtype: bool

.. method:: Bloomilter.clear_all()

    Removes all elements from the Bloom filter at once.

.. method:: BloomFilter.copy(filename)

    Copies the current :class:`BloomFilter` object to another object
    with a new filename.

    :param str filename: new filename
    :rtype: :class:`BloomFilter`

.. method:: BloomFilter.copy_template(filename, [perm=0755])

    Creates a new :class:`BloomFilter` object with the exact same parameters.
    Once this is performed, the two filters are comparable, so
    you can perform set operations using logical operators.

    Example::

        >>> apple = BloomFilter(100, 0.1, '/tmp/apple')
        >>> apple.add('granny_smith')
        False
        >>> pear = apple.copy_template('/tmp/pear')
        >>> pear.add('conference')
        False
        >>> pear |= apple

    :param str filename: new filename
    :param int perm: file access permission flags
    :rtype: :class:`BloomFilter`

.. method:: BloomFilter.sync()

    Forces a ``sync()`` call on the underlying mmap file object. Use this if
    you are about to copy the file and you want to be sure you got
    everything correctly.

.. method:: BloomFilter.to_base64()

    Serializes the :class:`BloomFilter` instance. Returns a compressed, base64 encoded string.
    This string can later be unpacked into a :class:`BloomFilter` using :meth:`BloomFilter.from_base64`.

    This may also be used to compare filter contents, given that the same ``error_rate``,
    ``capacity`` and ``hash_seeds`` were used when constructing such filters. For example::

        >>> b64_repr = "eJwFwUsOgjAUAMADuZCgKBsXhQeIWKRaEuquFihGPoYqDzm9M1U6LmUdU8UwUcNshM2IRssAwWfgSxjHjO6ssssn6bLsYTesqrtj0/dgYSuqzZ1cwISL1YrcH9V9PQ3cdN/JuRqn6nkRynUtd8rpmkldMt7Kb5EfF5d/IEl1GP/8LUuEYHN0HR5ihXL/1u65WKKZQkFsDykPfhQCpEAGGqexd4MX+vgkJ0/LCHIRNXpL0rk8SXH4A2pERcg="
        >>> hash_seeds = [3837895095, 3446164276, 218928576, 318812276, 2715048734, 4231234832, 2646234356, 1058991177, 1248068903, 1134013883, 3269341494, 3044656612, 3079736504]

        >>> bf = BloomFilter.from_base64("/tmp/bf", b64_repr)

        >>> bf_rec = BloomFilter(bf.capacity, bf.error_rate, "/tmp/bf_rec", hash_seeds=bf.hash_seeds.tolist())
        >>> bf_rec.add("5f35c4edcdb5b970ac8939a3c7abb3347ed9c4e3e251cbc799bdaeba008ce7aa")
        >>> bf_rec.add("f416d946d98166066611fb1a5e262c5f241d9bfdd8c885e062433b6f6b73799a")

        >>> assert bf_rec.to_base64() == bf.to_base64()

    :rtype: base64 encoded string representing filter

.. method:: BloomFilter.update(iterable)

    Calls :meth:`BloomFilter.add` on all items in the iterable.

.. method:: BloomFilter.union(filter)

    Performs a set OR with another comparable filter. You can (only) construct
    comparable filters with :meth:`BloomFilter.copy_template` above. In the above
    example, Bloom filter ``pear`` will have both "granny_smith" and "conference".

    The computation will occur *in place*. That is, calling::

        >>> bf.union(bf2)

    is a way of adding *all* the elements of ``bf2`` to ``bf``.

    *NB: Calling this function will render future calls to len()
    invalid.*

    :param BloomFilter other: filter to perform the union with
    :rtype: :class:`BloomFilter`

.. method:: BloomFilter.intersection(other)

    The same as :meth:`BloomFilter.union` above except it uses
    a set AND instead of a set OR.

    *NB: Calling this function will render future calls to len()
    invalid.*

    :param BloomFilter other: filter to perform the intersection with
    :rtype: :class:`BloomFilter`

.. method:: BloomFilter.close()

    Closes the currently opened :class:`BloomFilter` file descriptor.
    Following accesses to this instance will raise a :class:`ValueError`.

    *Caution*: this will delete an in-memory filter irrecoverably!


Magic Methods
-------------

.. method:: BloomFilter.__len__(item)

    Returns the number of distinct elements that have been
    added to the :class:`BloomFilter` object, subject to the error
    given in :attr:`BloomFilter.error_rate`.

    Example::

        >>> bf = BloomFilter(100, 0.1, '/tmp/fruit.bloom')
        >>> bf.add('apple')
        >>> bf.add('apple')
        >>> bf.add('orange')
        >>> len(bf)
        2

    Raises :class:`IndeterminateCountError` if a the Bloom filter
    was a result of a set operation. Example::

        >>> bf2 = bf.copy_template('/tmp/new.bloom')
        >>> bf2 |= bf
        >>> len(bf2)
        Traceback (most recent call last):
            ...
        pybloomfilter.IndeterminateCountError: Length of BloomFilter object is unavailable after intersection or union called.

    :param item: hashable object
    :rtype: int

.. method:: BloomFilter.__contains__(item)

    Checks to see if item is contained in the filter, with
    an acceptable false positive rate of :attr:`BloomFilter.error_rate` (see above).

    :param item: hashable object
    :rtype: bool

.. method:: BloomFilter.__ior__(filter)

    See :meth:`BloomFilter.union`.

.. method:: BloomFilter.__iand__(filter)

    See :meth:`BloomFilter.intersection`.


Exceptions
----------

.. class:: IndeterminateCountError(message)

    The exception that is raised if len() is called on a :class:`BloomFilter`
    object after \|=, &=, :meth:`BloomFilter.intersection`, or :meth:`BloomFilter.union` is used.
