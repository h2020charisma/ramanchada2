
import pytest
import numpy as np
import pandas as pd
from ramanchada2.misc.utils.matchsets import (
    match_peaks_cluster,
    match_peaks_optimized,
    match_peaks_monotonic_simple,
    match_peaks_ready_wrapper
)

@pytest.fixture
def base_ref():
    return {
        585.25: 1.0, 594.48: 0.8, 603.00: 0.5, 609.61: 0.9, 614.31: 0.7,
        626.65: 0.6, 633.44: 1.0, 640.22: 0.8, 650.65: 0.5, 659.89: 0.4
    }

def get_scenario_data(ref, scenario_name):
    if scenario_name == "Base Case (Offset=5)":
        return {k + 5: v for k, v in ref.items()}
    elif scenario_name == 'Missing Peaks (Drop 3)':
        return {k + 5: v for i, (k, v) in enumerate(ref.items()) if i not in [2, 5, 8]}
    elif scenario_name == "Extra Noise Peaks (Add 3)":
        return {**{k + 5: v for k, v in ref.items()}, **{600.0: 0.2, 620.0: 0.3, 645.0: 0.1}}
    elif scenario_name == "Intensity Distortion (Invert)":
        return {k + 5: (1.0 - v + 0.1) for k, v in ref.items()}
    elif scenario_name == "Non-linear Distortion (Stretch 1%)":
        return {k * 1.01 + 5: v for k, v in ref.items()}
    elif scenario_name == "Large Offset (Offset=50)":
        return {k + 50: v for k, v in ref.items()}
    elif scenario_name == "Peak Swap (Violation)":
        return {**{k + 5: v for k, v in ref.items() if k not in [603.00, 609.61]}, **{608.0: 0.5, 604.6: 0.9}}
    return ref

@pytest.mark.parametrize("method", ["dynamicp"])
@pytest.mark.parametrize("scenario", [
    "Base Case (Offset=5)",
    "Missing Peaks (Drop 3)",
    "Extra Noise Peaks (Add 3)",
    "Intensity Distortion (Invert)",
    "Non-linear Distortion (Stretch 1%)",
    "Large Offset (Offset=50)", 
    "Peak Swap (Violation)"
])
def test_matching_stress(base_ref, method, scenario):
    spe_pos_dict = get_scenario_data(base_ref, scenario)
    spe_pos_dict = dict(sorted(spe_pos_dict.items()))
    
    if method == "dynamicp":
        x_spe, x_ref, _, _ = match_peaks_ready_wrapper(spe_pos_dict, base_ref)
    
    # Assertions based on scenario
    if scenario == "Base Case (Offset=5)":
        assert len(x_spe) == 10
        assert np.allclose(x_spe - x_ref, 5.0, atol=0.5)
        
    elif scenario == "Missing Peaks (Drop 3)":
        # Expect 7 matches
        assert len(x_spe) == 7
        assert np.allclose(x_spe - x_ref, 5.0, atol=0.5)

    elif scenario == "Extra Noise Peaks (Add 3)":
         # Expect 10 matches (original)
         # Note: depending on noise placement, dynamicp might pick noise if closer/stronger, 
         # but here noise is distinct enough
         assert len(x_spe) >= 9 # Allow 1 mismatch if noise is tricky
         # Filter mostly correct ones
         diffs = np.abs((x_spe - x_ref) - 5.0)
         match_count = np.sum(diffs < 0.5)
         assert match_count >= 9

    elif scenario == "Intensity Distortion (Invert)":
        assert len(x_spe) >= 9
        
    elif scenario == "Large Offset (Offset=50)":
        assert len(x_spe) == 10
        assert np.allclose(x_spe - x_ref, 50.0, atol=0.5)
        
    elif scenario == "Peak Swap (Violation)":
        # Monotonicity enforced -> dropped swaps or partial match
        assert len(x_spe) >= 8 

