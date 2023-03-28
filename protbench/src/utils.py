import warnings
import functools


def mark_deprecated(use_instead: str = None):
    def _mark_deprecated(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            msg = f"`{func.__name__}` is deprecated and will be removed soon."
            if use_instead is not None:
                msg += f"Use `{use_instead}` instead."
            warnings.warn(msg, DeprecationWarning)
            out = func(*args, **kwargs)
            return out
        return wrapper
    return _mark_deprecated


def mark_experimental(use_instead: str = None):
    def _mark_experimental(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            msg = f"`{func.__name__}` is still experimental."
            if use_instead is not None:
                msg += f" Use `{use_instead}` for safety."
            warnings.warn(msg, UserWarning)
            out = func(*args, **kwargs)
            return out
        return wrapper
    return _mark_experimental
