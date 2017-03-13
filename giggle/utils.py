import functools
import itertools
import json
import traceback


def wrap_exceptions(func, logger=None):
    @functools.wraps(func)
    def func_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return _error_as_json(e, logger)
    return func_wrapper


def _error_as_json(ex, logger=None, status=500):
    if logger:
        logger.error(" -- Got exception in the tagger back-end!")
        logger.error(" -- %r" % ex)
        logger.error(traceback.format_exc())
    return json.dumps({'error': "{}".format(ex)}), status


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)
