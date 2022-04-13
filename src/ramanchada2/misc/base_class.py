class BaseClass(object):
    def __init__(self):
        self._origin = []

    def __repr__(self):
        return '/'.join(
            [i.replace('/', '_') for i in self.origin_list_str]
            ).replace(' ', '')

    @property
    def origin(self):
        return self._origin

    @origin.setter
    def origin(self, value):
        self._origin = list(value)

    @property
    def origin_list_str(self):
        ret = list()
        for name, args, kwargs in self.origin:
            str_args = [str(i) for i in args]
            str_kwargs = [f'{k}={repr(v)}' for k, v in kwargs.items()]
            sss = f'{name}({", ".join(str_args + str_kwargs)})'
            ret.append(sss)
        return ret
