# -*- coding: utf-8 -*- vim:fenc=utf-8:ft=python:et:sw=4:ts=4:sts=4
"""Tests (nose) for wor/bidirgencache module."""

import io
import random
import sys

import nose.tools as NT

from enum import Enum

from .bidirgencache import BidirGenCache

class SampleGenType(Enum):
    linear = 1
    looping = 2

def init_bidircachegen(
        sample_gen_type=SampleGenType.linear,
        prev_limit=-1, next_limit=-1,
        cache_prev_fn=None):
    def sample_gen_linear():
        for i in range(1, sys.maxsize):
            yield i
    def sample_gen_looping():
        for i in range(1, sys.maxsize):
            yield (i % 11)

    if sample_gen_type == SampleGenType.linear:
        generator = sample_gen_linear
    elif sample_gen_type == SampleGenType.looping:
        generator = sample_gen_looping
    else:
        assert(False)

    return BidirGenCache(
            generator(),
            prev_limit=prev_limit,
            next_limit=next_limit,
            cache_prev_fn=cache_prev_fn)

# -----------------

def test_bidirgencache_prev_noexception1():
    """Test BidirGenCache.prev()."""
    cgen = init_bidircachegen()
    next(cgen)
    cgen.prev(1)


@NT.raises(ValueError)
def test_bidirgencache_prev_exception1():
    """Test BidirGenCache.prev() ValueError exception."""
    init_bidircachegen().prev()


@NT.raises(ValueError)
def test_bidirgencache_prev_exception2():
    """Test BidirGenCache.prev() ValueError exception."""
    init_bidircachegen().prev(2)


@NT.raises(ValueError)
def test_bidirgencache_prev_exception3():
    """Test BidirGenCache.prev() ValueError exception."""
    cgen = init_bidircachegen()
    next(cgen)
    cgen.prev(2)

def test_bidirgencache_peek_noexception1():
    """Test BidirGenCache.peek()."""
    cgen = init_bidircachegen(SampleGenType.linear, -1, 2)
    cgen.peek(2)
    cgen.peek(1)

@NT.raises(ValueError)
def test_bidirgencache_peek_exception1():
    """Test BidirGenCache.peek() ValueError exception."""
    cgen = init_bidircachegen(SampleGenType.linear, -1, 2)
    cgen.peek(3)


def test_bidirgencache_prev_and_peek():
    """Test basic BidirGenCache functionality.

    This tests basic past and future caching of BidirGenCache generator wrapper.
    """
    cgen = init_bidircachegen()

    # Initial situation
    NT.eq_(cgen.prev_size(), 0)
    NT.eq_(cgen.next_size(), 0)
    NT.assert_true(cgen.is_next_first())

    # Get first value
    fst = next(cgen)

    # First value generated and cached to the prev cache
    NT.eq_(cgen.prev_size(), 1)
    NT.eq_(cgen.next_size(), 0)
    NT.eq_(cgen.prev(), fst)
    NT.assert_false(cgen.is_next_first())
    NT.eq_(list(cgen.get_prev_cache()), [fst])

    # Get the second value
    snd = next(cgen)

    # Second value generated and cached to the prev cache
    NT.eq_(cgen.prev_size(), 2)
    NT.eq_(cgen.next_size(), 0)
    NT.eq_(cgen.prev(1), snd)
    NT.eq_(cgen.prev(2), fst)
    NT.assert_false(cgen.is_next_first())
    NT.eq_(list(cgen.get_prev_cache()), [snd, fst])

    # Get the third value
    thd = next(cgen)
    NT.eq_(cgen.prev_size(), 3)
    NT.eq_(cgen.next_size(), 0)
    NT.eq_(cgen.prev(1), thd)
    NT.eq_(cgen.prev(2), snd)
    NT.eq_(cgen.prev(3), fst)
    NT.assert_false(cgen.is_next_first())
    NT.eq_(list(cgen.get_prev_cache()), [thd, snd, fst])

    # Test peeking
    fut_fourth = cgen.peek(1)

    NT.eq_(cgen.prev_size(), 3)
    NT.eq_(cgen.next_size(), 1)
    NT.eq_(cgen.prev(1), thd)
    NT.eq_(cgen.prev(2), snd)
    NT.eq_(cgen.prev(3), fst)
    NT.eq_(cgen.peek(), fut_fourth)
    NT.eq_(cgen.peek(1), fut_fourth)
    NT.eq_(list(cgen.get_next_cache()), [fut_fourth])

    fourth = next(cgen)

    NT.eq_(cgen.prev_size(), 4)
    NT.eq_(cgen.next_size(), 0)

    NT.eq_(fourth, fut_fourth)

    NT.eq_(cgen.prev(1), fourth)
    NT.eq_(cgen.prev(2), thd)
    NT.eq_(cgen.prev(3), snd)
    NT.eq_(cgen.prev(4), fst)
    NT.eq_(list(cgen.get_prev_cache()), [fourth, thd, snd, fst])

    # Peek more
    fut_6th = cgen.peek(2)

    NT.eq_(cgen.prev_size(), 4)
    NT.eq_(cgen.next_size(), 2)
    NT.eq_(cgen.prev(1), fourth)
    NT.eq_(cgen.prev(2), thd)
    NT.eq_(cgen.prev(3), snd)
    NT.eq_(cgen.prev(4), fst)
    NT.eq_(cgen.peek(2), fut_6th)
    NT.eq_(len(list(cgen.get_next_cache())), 2)

    fifth = next(cgen)

    NT.eq_(cgen.prev_size(), 5)
    NT.eq_(cgen.next_size(), 1)
    NT.eq_(cgen.prev(1), fifth)
    NT.eq_(cgen.prev(2), fourth)
    NT.eq_(cgen.prev(3), thd)
    NT.eq_(cgen.prev(4), snd)
    NT.eq_(cgen.prev(5), fst)
    NT.eq_(cgen.peek(), fut_6th)
    NT.eq_(cgen.peek(1), fut_6th)
    NT.eq_(list(cgen.get_prev_cache()), [fifth, fourth, thd, snd, fst])

    sixth = next(cgen)

    NT.eq_(cgen.prev_size(), 6)
    NT.eq_(cgen.next_size(), 0)
    NT.eq_(fut_6th, sixth)
    NT.eq_(cgen.prev(1), sixth)
    NT.eq_(cgen.prev(2), fifth)
    NT.eq_(cgen.prev(3), fourth)
    NT.eq_(cgen.prev(4), thd)
    NT.eq_(cgen.prev(5), snd)
    NT.eq_(cgen.prev(6), fst)
    NT.eq_(list(cgen.get_prev_cache()), [sixth, fifth, fourth, thd, snd, fst])

def test_bidirgencache_caches():
    cgen = init_bidircachegen()
    cur_1 = next(cgen)
    cur_2 = next(cgen)
    cur_3 = next(cgen)
    fut_1 = cgen.peek(1)
    fut_2 = cgen.peek(2)
    fut_3 = cgen.peek(3)

    NT.eq_(cgen.prev_size(), 3)
    NT.eq_(cgen.next_size(), 3)

    NT.eq_(list(cgen.get_prev_cache()), [cur_3, cur_2, cur_1])
    NT.eq_(list(cgen.get_next_cache()), [fut_1, fut_2, fut_3])

def test_bidirgencache_send1():
    """Test BidirGenCache.send()."""
    cgen = init_bidircachegen()

    # Peek 3 values
    fut_1st = cgen.peek(1)
    NT.eq_(cgen.next_size(), 1)
    NT.eq_(cgen.prev_size(), 0)
    fut_2nd = cgen.peek(2)
    NT.eq_(cgen.next_size(), 2)
    fut_3th = cgen.peek(3)
    NT.eq_(cgen.next_size(), 3)
    NT.eq_(cgen.prev_size(), 0)

    # Get first value, which should be peek(1)
    cur_1st = next(cgen)
    NT.eq_(cur_1st, fut_1st)

    NT.eq_(cgen.next_size(), 2)
    NT.eq_(cgen.prev_size(), 1)

    send_value = 999
    cgen.send(send_value)

    NT.eq_(cgen.next_size(), 3)
    NT.eq_(cgen.prev_size(), 1)

    # Get second value, which should be the sended value
    cur_2nd = next(cgen)

    NT.eq_(cur_2nd, send_value)

    NT.eq_(cgen.next_size(), 2)
    NT.eq_(cgen.prev_size(), 2)

    # Get third value, which should be the fut_2nd
    cur_3th = next(cgen)

    NT.eq_(cur_3th, fut_2nd)

    NT.eq_(cgen.next_size(), 1)
    NT.eq_(cgen.prev_size(), 3)

    # Get fourth value, which should be the fut_3th
    cur_4th = next(cgen)

    NT.eq_(cur_4th, fut_3th)

    NT.eq_(cgen.next_size(), 0)
    NT.eq_(cgen.prev_size(), 4)

def test_bidirgencache_send_noexception1():
    """Test BidirGenCache.send()."""
    cgen = init_bidircachegen(SampleGenType.linear, -1, 1)
    cgen.send(1)

@NT.raises(KeyError)
def test_bidirgencache_send_exception1():
    """Test BidirGenCache.send()."""
    cgen = init_bidircachegen(SampleGenType.linear, -1, 1)
    cgen.send(1)
    cgen.send(2)

def test_bidirgencache_func_cache1():
    """Test BidirGenCache with cache_prev_fn."""
    def prev_test(value):
        return value == 5
    cgen = init_bidircachegen(SampleGenType.looping, cache_prev_fn=prev_test)

    #print()
    NT.eq_(cgen.prev_size(), 0)
    for i in range(1, 5):
        next(cgen)
        #print(cgen._past_cache)
        NT.eq_(cgen.prev_size(), i)

    next(cgen)
    NT.eq_(cgen.prev_size(), 1)

    for i in range(2, 12):
        next(cgen)
        #print(cgen._past_cache)
        NT.eq_(cgen.prev_size(), i)

    next(cgen)
    NT.eq_(cgen.prev_size(), 1)
