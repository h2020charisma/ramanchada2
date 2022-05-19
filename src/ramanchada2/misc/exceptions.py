#!/usr/bin/env python3


class ApplicationException(Exception):
    pass


class InputParserError(ApplicationException):
    pass


class ChadaReadNotFoundError(ApplicationException):
    pass
