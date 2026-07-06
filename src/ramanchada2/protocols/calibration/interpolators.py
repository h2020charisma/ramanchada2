from scipy.interpolate import CubicSpline, PchipInterpolator, RBFInterpolator
import numpy as np
import json
from typing import Literal

# Single source of truth for the interpolator options offered across the calibration API.
# "pchippolyinverse" is a deprecated alias of "polyinverse", accepted for backward
# compatibility. "rbf" and "cubic_spline" were removed from the public options;
# their classes are kept so previously saved models still load.
InterpolatorMethod = Literal[
    "poly", "polyinverse", "pchip", "pchipinverse", "rbfinverse", "pchippolyinverse"
]


class CustomPChipInterpolator(PchipInterpolator):
    def __init__(self, x, y,  inverse=False, **kwargs):
        if inverse:
            inverse = PchipInterpolator(y, x)
            dense_reference = np.linspace(
                y[0],
                y[-1],
                max(2048, 10*len(y)) # N≫number of original knots
                # Anything smaller gives you no benefit over the raw calibration.
            )
            dense_spe = inverse(dense_reference)
            order = np.argsort(dense_spe)
            dense_spe = dense_spe[order]
            dense_reference = dense_reference[order]  
            super().__init__(dense_spe, dense_reference,  **kwargs)
            self.x = dense_spe  # Store x values
            self.y = dense_reference  # Store y values
            self.x_original = x  # Store x values
            self.y_original = y  # Store y values                  
        else:
            # direct        
            super().__init__(x, y,  **kwargs)
            self.x = x  # Store x values
            self.y = y  # Store y values
        self.inverse = inverse

    def __call__(self, x, nu=0, extrapolate=None):
        """Evaluate; beyond the knot span continue with CONSTANT CORRECTION.

        The PCHIP edge cubic is unconstrained outside the anchors (Ne lines end well below
        the CH-stretch wavelength at both lasers) and can run away by tens of nm, corrupting
        the ~2900 cm-1 region and the Si laser-zeroing peak when they fall outside the span.
        Extending the edge *correction* (unit slope) is the physically safe extrapolation
        for a wavelength-correction map.
        """
        out = super().__call__(x, nu=nu, extrapolate=extrapolate)
        if nu != 0:
            return out
        knots = np.asarray(self.x, dtype=float)
        lo, hi = knots[0], knots[-1]
        x_arr = np.asarray(x, dtype=float)
        if x_arr.size == 0 or (x_arr.min() >= lo and x_arr.max() <= hi):
            return out
        y_lo = float(super().__call__(lo))
        y_hi = float(super().__call__(hi))
        out = np.asarray(out, dtype=float)
        out = np.where(x_arr < lo, y_lo + (x_arr - lo), out)
        out = np.where(x_arr > hi, y_hi + (x_arr - hi), out)
        return out

    @staticmethod
    def from_dict(pchip_dict=None):
        if pchip_dict is None:
            pchip_dict = {}
        # Load the PCHIP interpolator from a dictionary
        interpolator_loaded = CustomPChipInterpolator(
            np.array(pchip_dict["x"]),  # Convert back to numpy arrays
            np.array(pchip_dict["y"]),
        )
        # restore anchor provenance saved by inverse variants (optional keys)
        if pchip_dict.get("x_original") is not None:
            interpolator_loaded.x_original = np.array(pchip_dict["x_original"])
            interpolator_loaded.y_original = np.array(pchip_dict["y_original"])
        return interpolator_loaded

    def to_dict(self):
        # Save the current x and y data to a dictionary
        d = {
            "x": np.asarray(self.x).tolist(),  # Convert numpy arrays to lists for JSON serialization
            "y": np.asarray(self.y).tolist(),
        }
        # inverse variants keep the original anchor knots separately from the dense grid;
        # preserve them so span/provenance diagnostics survive a JSON round-trip
        if getattr(self, "x_original", None) is not None:
            d["x_original"] = np.asarray(self.x_original).tolist()
            d["y_original"] = np.asarray(self.y_original).tolist()
        return d

    def save_coefficients(self, filename):
        """Save the x and y coefficients to a JSON file."""
        coeffs = self.to_dict()
        with open(filename, "w") as f:
            json.dump(coeffs, f)

    @classmethod
    def load_coefficients(cls, filename):
        """Load the coefficients from a JSON file."""
        with open(filename, "r") as f:
            coeffs = json.load(f)
        return cls.from_dict(coeffs)

    def plot(self, ax):
        """Plot the interpolation curve and the original points."""
        ax.scatter(self.x, self.y, marker="+", color="blue", label="Ne original data")

        x_range = np.linspace(self.x.min(), self.x.max(), 100)
        predicted_y = self(x_range)

        ax.plot(
            x_range, predicted_y, color="red", linestyle="-", label="Ne calibration curve"
        )
        ax.set_xlabel("Ne peak original/nm")
        ax.set_ylabel("Ne peak NIST/nm")
        ax.grid(which="both", linestyle="--", linewidth=0.5, color="gray")
        ax.legend()

    def __str__(self):
        return f"Calibration curve {len(self.y)} points) (PchipInterpolator)"


class CustomCubicSplineInterpolator(CubicSpline):
    def __init__(self, x, y,  **kwargs):
        super().__init__(x, y, **kwargs)
        self.x = x
        self.y = y

    @staticmethod
    def from_dict(spline_dict=None):
        if spline_dict is None:
            spline_dict = {}
        interpolator_loaded = CustomCubicSplineInterpolator(
            spline_dict["x"],
            spline_dict["y"],
            bc_type=spline_dict.get("bc_type", "clamped"),
            extrapolate=spline_dict.get("extrapolate", True),
        )
        return interpolator_loaded

    def to_dict(self):
        return {
            "x": np.asarray(self.x).tolist(),
            "y": np.asarray(self.y).tolist(),
            "bc_type": self.bc_type,
            "extrapolate": bool(self.extrapolate),
        }

    def plot(self, ax):
        ax.scatter(self.x, self.y, marker="+", color="blue", label="Data points")
        x_range = np.linspace(self.x.min(), self.x.max(), 100)
        predicted_y = self(x_range)

        ax.plot(
            x_range, predicted_y, color="red", linestyle="-", label="Cubic spline curve"
        )
        ax.set_xlabel("X values")
        ax.set_ylabel("Y values")
        ax.grid(which="both", linestyle="--", linewidth=0.5, color="gray")
        ax.legend()

    def __str__(self):
        return f"Cubic Spline Interpolator with {len(self.x)} points."


class CustomPolyInterpolator:
    def __init__(self, x, y, inverse=False, max_degree=4):
        if inverse:
            inverse = PchipInterpolator(y, x)
            dense_reference = np.linspace(
                y[0],
                y[-1],
                max(2048, 10*len(y)) # N≫number of original knots
                # Anything smaller gives you no benefit over the raw calibration.
            )
            dense_spe = inverse(dense_reference)
            order = np.argsort(dense_spe)
            dense_spe = dense_spe[order]
            dense_reference = dense_reference[order]  
            self.x = dense_spe  # Store x values
            self.y = dense_reference  # Store y values
            self.x_original = x  # Store x values
            self.y_original = y  # Store y values                  
        else:
            # direct        
            self.x = x  # Store x values
            self.y = y  # Store y values
        self.inverse = inverse
        self.max_degree = max_degree

        # enforce monotonic ordering in x
        order = np.argsort(self.x)
        self.x = self.x[order]
        self.y = self.y[order]

        # normalize x for numerical stability
        self.x_min = self.x.min()
        self.x_max = self.x.max()

        u = (self.x - self.x_min) / (self.x_max - self.x_min)

        # choose degree ≤ max_degree by max residual, PARSIMONIOUSLY: a higher degree is
        # accepted only when it reduces the max residual substantially (>20%). Plain
        # minimization always rewards extra degrees, so noisy anchor sets (blended Ne
        # lines, ±0.5 nm center scatter) get a noise-chasing fit whose edge tilt is then
        # amplified over the whole cm-1 axis by the Si laser-zeroing.
        best_err = np.inf
        best_coeff = None
        best_deg = None

        for deg in range(2, max_degree + 1):
            coeff = np.polyfit(u, self.y, deg)
            pred = np.polyval(coeff, u)
            err = np.max(np.abs(pred - self.y))

            if err < 0.8 * best_err:
                best_err = err
                best_coeff = coeff
                best_deg = deg

        self.coef = best_coeff
        self.degree = best_deg
        self.fit_error = best_err

    # --------------------------------------------------------
    # Callable interface (like PchipInterpolator)
    # --------------------------------------------------------
    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        u = (x - self.x_min) / (self.x_max - self.x_min)
        out = np.polyval(self.coef, u)
        # Beyond the anchor span the polynomial tail is unconstrained and can run away by
        # tens of nm (Ne lines end well below the CH-stretch wavelength at both lasers).
        # Continue with CONSTANT CORRECTION (unit slope) from the edge value instead --
        # the physically safe extrapolation for a wavelength-correction map.
        if x.size and (x.min() < self.x_min or x.max() > self.x_max):
            y_lo = np.polyval(self.coef, 0.0)
            y_hi = np.polyval(self.coef, 1.0)
            out = np.where(x < self.x_min, y_lo + (x - self.x_min), out)
            out = np.where(x > self.x_max, y_hi + (x - self.x_max), out)
        return out

    # --------------------------------------------------------
    # Serialization
    # --------------------------------------------------------
    @staticmethod
    def from_dict(poly_dict=None):
        if poly_dict is None:
            poly_dict = {}

        obj = CustomPolyInterpolator(
            np.array(poly_dict["x"]),
            np.array(poly_dict["y"]),
            max_degree=poly_dict.get("degree", 3),
        )

        # overwrite learned parameters
        obj.coef = np.array(poly_dict["coef"])
        obj.degree = poly_dict["degree"]
        obj.x_min = poly_dict["x_min"]
        obj.x_max = poly_dict["x_max"]
        obj.fit_error = poly_dict.get("fit_error", None)
        # restore anchor provenance saved by inverse variants (optional keys)
        if poly_dict.get("x_original") is not None:
            obj.x_original = np.array(poly_dict["x_original"])
            obj.y_original = np.array(poly_dict["y_original"])

        return obj

    def to_dict(self):
        d = {
            "x": np.asarray(self.x).tolist(),
            "y": np.asarray(self.y).tolist(),
            "coef": np.asarray(self.coef).tolist(),
            "degree": int(self.degree),
            "x_min": float(self.x_min),
            "x_max": float(self.x_max),
            "fit_error": None if self.fit_error is None else float(self.fit_error),
        }
        if getattr(self, "x_original", None) is not None:
            d["x_original"] = np.asarray(self.x_original).tolist()
            d["y_original"] = np.asarray(self.y_original).tolist()
        return d

    def save_coefficients(self, filename):
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load_coefficients(cls, filename):
        with open(filename, "r") as f:
            coeffs = json.load(f)
        return cls.from_dict(coeffs)

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    def plot(self, ax):
        ax.scatter(self.x, self.y, marker="+", color="blue", label="Original data")

        x_range = np.linspace(self.x.min(), self.x.max(), 500)
        y_pred = self(x_range)

        ax.plot(
            x_range,
            y_pred,
            color="red",
            linestyle="-",
            label=f"Polynomial degree {self.degree}",
        )
        ax.set_xlabel("Original")
        ax.set_ylabel("Reference")
        ax.grid(which="both", linestyle="--", linewidth=0.5, color="gray")
        ax.legend()

    def __str__(self):
        return (
            f"Calibration curve {len(self.y)} points "
            f"(Polynomial degree {self.degree})"
        )


class CustomRBFInterpolator(RBFInterpolator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_dict(rbf_dict=None):
        if rbf_dict is None:
            rbf_dict = {}
        interpolator_loaded = CustomRBFInterpolator(
            rbf_dict["y"],
            rbf_dict["d"],
            epsilon=rbf_dict["epsilon"],
            smoothing=rbf_dict["smoothing"],
            kernel=rbf_dict["kernel"],
            neighbors=rbf_dict["neighbors"],
        )
        interpolator_loaded._coeffs = rbf_dict["coeffs"]
        interpolator_loaded._scale = rbf_dict["scale"]
        interpolator_loaded._shift = rbf_dict["shift"]
        return interpolator_loaded

    def to_dict(self):
        return {
            "y": self.y,
            "d": self.d,
            "d_dtype": self.d_dtype,
            "d_shape": self.d_shape,
            "epsilon": self.epsilon,
            "kernel": self.kernel,
            "neighbors": self.neighbors,
            "powers": self.powers,
            "smoothing": self.smoothing,
            "coeffs": self._coeffs,
            "scale": self._scale,
            "shift": self._shift,
        }

    def plot(self, ax):
        ax.scatter(
            self.y.reshape(-1),
            self.d.reshape(-1),
            marker="+",
            color="blue",
            label="Matched peaks",
        )

        x_range = np.linspace(self.y.min(), self.y.max(), 100)
        predicted_x = self(x_range.reshape(-1, 1))

        ax.plot(
            x_range, predicted_x, color="red", linestyle="-", label="Calibration curve"
        )
        ax.set_xlabel("Ne peaks, nm")
        ax.set_ylabel("Reference peaks, nm")
        ax.grid(which="both", linestyle="--", linewidth=0.5, color="gray")
        ax.legend()

    def __str__(self):
        return f"Calibration curve {len(self.y)} points) {self.kernel}"


def get_interpolator(x_spe, x_reference, interpolator_method: InterpolatorMethod = "poly"):
    if interpolator_method == "pchip":
        interp = CustomPChipInterpolator(x_spe, x_reference, inverse=False)
    elif interpolator_method == "pchipinverse":
        interp = CustomPChipInterpolator(x_spe, x_reference, inverse=True)
    elif interpolator_method in ["polyinverse", "pchippolyinverse"]:
        interp = CustomPolyInterpolator(x_spe, x_reference, inverse=True)
    elif interpolator_method in ["poly"]:
        interp = CustomPolyInterpolator(x_spe, x_reference, inverse=False)
    elif interpolator_method == "rbfinverse":
        # CWA / MATLAB recipe (phspline + interp1 'pchip'): the polyharmonic (thin-plate)
        # spline supplies only a SMOOTH SHAPE, evaluated forward (reference -> measured) on a
        # dense, sorted reference grid; the monotone measured -> reference calibration map is
        # then a PCHIP through those dense samples. PCHIP -- not the RBF -- guarantees
        # monotonicity, so the calibrated axis cannot fold (a raw forward RBF can).
        x_spe = np.asarray(x_spe, dtype=float)
        x_reference = np.asarray(x_reference, dtype=float)
        fwd = RBFInterpolator(
            x_reference.reshape(-1, 1), x_spe,
            kernel="thin_plate_spline", smoothing=0,
        )
        dense_ref = np.linspace(
            x_reference.min(), x_reference.max(), max(2048, 10 * len(x_reference))
        )
        dense_spe = np.asarray(fwd(dense_ref.reshape(-1, 1))).reshape(-1)
        order = np.argsort(dense_spe)
        interp = CustomPChipInterpolator(dense_spe[order], dense_ref[order])
    else:
        raise Exception(f"Unknown interpolator {interpolator_method}")
    return interp


# Registry for portable (JSON) serialization: class name <-> class. CustomRBFInterpolator is
# deliberately absent -- its to_dict is not JSON-clean and "rbfinverse" already produces a
# CustomPChipInterpolator, so no persisted model needs it.
INTERPOLATOR_CLASSES = {
    "CustomPChipInterpolator": CustomPChipInterpolator,
    "CustomPolyInterpolator": CustomPolyInterpolator,
    "CustomCubicSplineInterpolator": CustomCubicSplineInterpolator,
}


def interpolator_to_tagged_dict(interp):
    """Serialize an interpolator to a JSON-clean dict with a "type" tag."""
    name = type(interp).__name__
    if name not in INTERPOLATOR_CLASSES:
        raise NotImplementedError(
            f"{name} has no portable JSON serialization; re-derive the model with a "
            "supported interpolator_method")
    return {"type": name, **interp.to_dict()}


def interpolator_from_tagged_dict(d):
    """Reconstruct an interpolator from a dict produced by interpolator_to_tagged_dict."""
    cls = INTERPOLATOR_CLASSES.get(d.get("type"))
    if cls is None:
        raise ValueError(f"Unknown interpolator type {d.get('type')!r}")
    payload = {k: v for k, v in d.items() if k != "type"}
    return cls.from_dict(payload)
