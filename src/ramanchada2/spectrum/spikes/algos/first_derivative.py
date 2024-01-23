import numpy as np
def first_derivative(intensity, threshold):
    # Using second order central differences for Laplacian
    # The Laplacian of a 1D function is its second derivative
    laplacian = np.diff(intensity, prepend = [0])    
    return laplacian