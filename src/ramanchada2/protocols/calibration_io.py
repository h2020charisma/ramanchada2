# serialization.py
import json
import numpy as np
from scipy.interpolate import RBFInterpolator
from ramanchada2.protocols.calibration import XCalibrationComponent 
from ramanchada2.protocols.calibration_deco import add_calibration_serialisation

@add_calibration_serialisation(XCalibrationComponent)
def to_json(self):
    return xcalibration_model_to_json(self)


def ndarray_to_list(arr):
    return arr.tolist()

def list_to_ndarray(lst):
    return np.array(lst)



def xcalibration_model_to_json( 
            xcal : XCalibrationComponent
            ):
    if other_data is None:
        other_data = {}

    # Extracting data from the RBFInterpolator object
    data_to_save = {
        'laser_wl' : xcal.laser_wl,
        'model' : {
            'x': ndarray_to_list(xcal.model.x),
            'y': ndarray_to_list(xcal.model.y),
            'epsilon': xcal.model.epsilon,
            'smoothing': xcal.model.smoothing,
            'kernel': xcal.model.kernel,
            'coefficients': ndarray_to_list(xcal.model._coeffs),
            },
        'spe_neon' : {
            'units' : xcal.spe_neon_units
        },
        'peaks_reference' : {
            'units' : xcal.ref_neon_units
        }
    }
    return data_to_save


def load_rbf_interpolator(filepath):
    # Load from JSON file
    with open(filepath, 'r') as file:
        data_loaded = json.load(file)

    laser_wl = data_loaded["laser_wl"]
    with data_loaded['model'] as model:
        x_loaded = list_to_ndarray(model['x'])
        y_loaded = list_to_ndarray(model['y'])
        epsilon_loaded = model['epsilon']
        smoothing_loaded = model['smoothing']
        kernel_loaded = model['kernel']
        coefficients_loaded = list_to_ndarray(model['coefficients'])
    other_data_loaded = data_loaded.get('other_data', {})

    # Custom class to load RBFInterpolator with coefficients
    class CustomRBFInterpolator(RBFInterpolator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def set_coefficients(self, coeffs):
            self._coeffs = coeffs

    # Create the RBFInterpolator object
    # Map the kernel name back to the actual function
    kernel_function = getattr(RBFInterpolator, kernel_loaded)

    # Create the RBFInterpolator object
    rbf_loaded = CustomRBFInterpolator(x_loaded, y_loaded, epsilon=epsilon_loaded, smoothing=smoothing_loaded, kernel=kernel_function)
    rbf_loaded.set_coefficients(coefficients_loaded)

    return rbf_loaded, other_data_loaded
