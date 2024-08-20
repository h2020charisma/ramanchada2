
from functools import wraps
from .calibration import XCalibrationComponent

def add_calibration_serialisation(cls):
    def decorator(fun):
        @wraps(fun)
        def retf(obj, *args, **kwargs):
            ret = fun(obj, *args, **kwargs)
            return ret
        setattr(cls, fun.__name__, retf)
        print(f"Method {fun.__name__} added to {cls.__name__}")
        return retf
    return decorator
