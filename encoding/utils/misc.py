import warnings

__all__ = ['EncodingDeprecationWarning']

class EncodingDeprecationWarning(DeprecationWarning):
    pass

warnings.simplefilter('once', EncodingDeprecationWarning)
