# -*- coding: utf-8 -*- vim:fenc=utf-8:ft=python:et:sw=4:ts=4:sts=4
"""BidirGenCache generator wrapper with caching.
"""
import collections

class BidirGenCache(object):
    """Generator wrapper which caches past values and allows peeking.

    It's basically a memoryview for a generator.

    Other functionality mainly targeted at token stream generation includes
    dynamic size previous caching using 'cache_prev_fn' and the generated
    objects attribute modifier using 'attr_mod' dictionary.

    Attributes:
        prev_limit: int. How many previous variables are cached as integer. "-1"
            has special meaning to cache as long as memory runs out.
        next_limit: int. How many future variables are cached. If this is 3 then
            peek(3) works but peek(4) or larger doesn't. "-1" has special
            meaning to allow peeking until memory is exhausted or wrapped
            generator ends.
        attr_mod: {str->obj}. See __init__.

    TODO:
        complete doc strings
    """
    def __init__(self, generator, prev_limit=1, next_limit=1,
            cache_prev_fn=None, attr_mod={}):
        """Initilizes bidirectional generator cacher.

        Args:
            generator: yieldable. Generator which is wrapped.
            prev_limit: see BidirGenCache prev_limit attribute.
            next_limit: see BidirGenCache next_limit attribute.
            cache_prev_fn: callable(object). Function which takes one parameter
                and returns True if generator produced value breaks caching. This
                means that actual number of objects in prev cache varies as it's
                emptied when this function returns True for the object yielded by
                the generator. This is a way to hold context dependant number of
                objects in the cache.
            attr_mod: {str->obj}. Mapping from attribute name to attribute
                value. The object generated from the wrapped generator will have
                these attributes set. This happens also for sent objects using send().
                There's a condition that the attributes must already exist and
                have the value of 'None'.

        Raises:
            ValueError: If invalid prev_limit or next_limit are given. Meaning values < -1.
        """
        self._cache_prev_fn = cache_prev_fn
        if self._cache_prev_fn != None:
            prev_limit = -1
        self.__next_limit = next_limit
        self.__prev_limit = prev_limit
        self._generator = generator
        # Cache contains past (previous) values
        self._past_cache = collections.deque(maxlen=prev_limit if prev_limit != -1 else None)
        # Future cache contains future (next) values
        self._future_cache = collections.deque(maxlen=next_limit if next_limit != -1 else None)
        self.attr_mod = attr_mod
    def _preform_attr_mod(self, obj):
        """Modifies given objects attributes to match 'attr_mod' mapping.

        Object 'obj' must have the attribute already and it's value must be 'None'.

        Args:
            obj. object: Object which attributes will be modified.
        """
        for key, value in self.attr_mod.items():
            try:
                attr = getattr(obj, key)
                if attr == None:
                    setattr(obj, key, value)
            except AttributeError:
                pass
    def _get_next(self):
        """Next call wrapped with attribute modifying."""
        retval = next(self._generator)
        self._preform_attr_mod(retval)
        return retval
    def __next__(self):
        """Next value getter for iteration.

        Next value from future cache or if it's empty then from the wrapped
        generator.
        """
        def is_past_cache_full():
            """Test if past cache is full and clear cache if necessary."""
            if callable(self._cache_prev_fn):
                if self._cache_prev_fn(retval):
                    self._past_cache.clear()
                return False
            return len(self._past_cache) == self.prev_limit

        if len(self._future_cache):
            retval = self._future_cache.popleft()
            # Move value from future to past cache
            # First check if past cache full
            if is_past_cache_full():
                self._past_cache.popleft()
            self._past_cache.append(retval)
        else:
            retval = self._get_next()
            if is_past_cache_full():
                self._past_cache.popleft()
            self._past_cache.append(retval)
        return retval
    def __iter__(self):
        return self
    def __str__(self):
        return "{}/{}, {}/{}, {}, {}".format(
                len(self._past_cache), self.prev_limit, len(self._future_cache),
                self.next_limit, self._past_cache, self._future_cache)
    def _fill_cache(self, cache, count):
        for _ in range(0,count-len(cache)):
            cache.append(self._get_next())
    ### Read only properties:
    @property
    def next_limit(self):
        """Size limit of the next/future cache."""
        return self.__next_limit
    @property
    def prev_limit(self):
        """Size limit of the prev/past cache."""
        return self.__prev_limit
    ###
    def next_size(self):
        """Size of the next/future cache."""
        return len(self._future_cache)
    def prev_size(self):
        """Size of the prev/past cache."""
        return len(self._past_cache)
    def is_next_first(self):
        """Has next been called previously.

        Returns None if it cannot be determined using past cache, meaning past
        cache size is set to 0.
        """
        return None if (self.prev_limit == 0) else (len(self._past_cache) == 0)
    def peek(self, count=1):
        """Peek next value(s).

        If not enough cached values are present then they are generated by
        calling next for the encompassed generator.

        Args:
            count. uint. Valid values in range(1, next_limit)

        Raises:
            ValueError: If invalid count value given (1 > count > next_limit).
            Max limit only if next_limit is not "-1", meaning future/peek cache
            is not limited.
            StopIteration: Raised possibly by the wrapped generator.
        """
        if count < 1:
            raise ValueError("Invalid peek count '{}'".format(count))
        if self.next_limit != -1 and count > self.next_limit:
            raise ValueError("Future/next cache not large enough '{} < {}'".format(self.next_limit, count))

        # TODO: check what happens if future cache limit is met
        if len(self._future_cache) < count:
            # Future value(s) not already cached, call next for the wrapped
            # generator
            self._fill_cache(self._future_cache, count)
        return self._future_cache[count-1]
    def prev(self, count=1):
        """Returns cached values from the past/prev cache.

        Args:
            count. uint. Valid values in range(1, prev_size).

        Raises:
            ValueError: If invalid count value given (1 > count > prev_size).
        """
        if count < 1:
            raise ValueError("Invalid prev count '{}'".format(count))
        if count > len(self._past_cache):
            raise ValueError("Prev cache not large enough '{} < {}'".format(len(self._past_cache), count))

        return self._past_cache[-count]
    def send(self, item):
        """Sends item back to generator (cache).

        The next generated value is then the last sended value. This means that
        the sended value will be peek(1).

        Sending values needs space from the next/future cache.

        Args:
            item. object. Item to be added to the future cache.

        Raises:
            KeyError: If future value cache is full.
        """
        if self.next_limit != -1 and self.next_limit <= len(self._future_cache):
            raise KeyError("Generator cache full.")
        else:
            self._preform_attr_mod(item)
            self._future_cache.appendleft(item)
    def get_prev_cache(self):
        """Returns a generator for the prev/past cache.

        First value generated by this is the most recent value generated by the
        wrapped generator.
        """
        def prev_cache_gen():
            for i in reversed(self._past_cache):
                yield i
        return prev_cache_gen()
    def get_next_cache(self):
        """Returns a generator for the next/future cache."""
        def future_cache_gen():
            for i in self._future_cache:
                yield i
        return future_cache_gen()
