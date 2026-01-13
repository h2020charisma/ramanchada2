import numpy as np


def shift_cm_1_to_abs_nm_dict(deltas, laser_wave_length_nm):
    arr = np.array(list(deltas.items()), dtype=float)
    arr[:, 0] = 1/(1/laser_wave_length_nm - arr[:, 0] * 1e-7)
    return dict(arr)


def abs_nm_to_shift_cm_1_dict(deltas, laser_wave_length_nm):
    arr = np.array(list(deltas.items()), dtype=float)
    arr[:, 0] = 1e7*(1/laser_wave_length_nm - 1/arr[:, 0])
    return dict(arr)


def abs_nm_to_shift_cm_1(wl, laser_wave_length_nm):
    return 1e7*(1/laser_wave_length_nm - 1/wl)


def shift_cm_1_to_abs_nm(wn, laser_wave_length_nm):
    shift_nm = wn * 1e-7
    absolute_nm = 1/(1/laser_wave_length_nm - shift_nm)
    return absolute_nm


def laser_wl_nm(raman_shift_cm_1, wave_length_nm):
    return 1/(1/wave_length_nm+raman_shift_cm_1*1e-7)


def filter_ref_lines_for_raman(
    ref_wavelengths,
    laser_wl_nm,
    raman_shift_range_cm_1=(-500, 4000),
    margin_nm=1
):
    """
    Filter Ne reference lines to expected Raman detection range.
    
    Args:
        ne_wavelengths: Full NIST Ne line list (nm)
        laser_wl_nm: Laser wavelength (nm)
        raman_shift_range_cm_1: (min, max) Raman shift in cm⁻¹
        margin_nm: Extra wavelength margin (nm)
    
    Returns:
        Filtered Reference wavelengths array
    """
    # Convert Raman shift range to wavelength range
    wl_min = shift_cm_1_to_abs_nm(raman_shift_range_cm_1[1], laser_wl_nm) - margin_nm
    wl_max = shift_cm_1_to_abs_nm(raman_shift_range_cm_1[0], laser_wl_nm) + margin_nm
    
    wl_min, wl_max = sorted([wl_min, wl_max])
    # Filter
    ne_array = np.asarray(ref_wavelengths)
    mask = (ne_array >= wl_min) & (ne_array <= wl_max)
    filtered = ne_array[mask]
    
    print(f"Laser: {laser_wl_nm} nm")
    print(f"Raman range: {raman_shift_range_cm_1[0]} to {raman_shift_range_cm_1[1]} cm⁻¹")
    print(f"Wavelength range: {wl_min:.1f} - {wl_max:.1f} nm")
    print(f"Reference lines: {len(ref_wavelengths)} total → {len(filtered)} in range")
    
    return filtered


# Helper function to create filtered dicts
def _create_dict_for_laser(laser_wl_nm, raman_range=(-500, 4000), peaks=None, margin_nm=10):
    """
    Create Neon wavelength dict for a specific laser by filtering ne_peaks_cwa.
    
    Returns dict of {wavelength_nm: intensity_placeholder}
    """
    from ramanchada2.misc.utils.ramanshift_to_wavelength import (
        shift_cm_1_to_abs_nm, filter_ref_lines_for_raman
    )
    
    # Filter wavelengths for this laser
    filtered_wl = filter_ref_lines_for_raman(
        ref_wavelengths=peaks,
        laser_wl_nm=laser_wl_nm,
        raman_shift_range_cm_1=raman_range,
        margin_nm=margin_nm
    )
    
    # Create dict with placeholder intensities (or use 1.0 as default)
    # If you want actual intensities, you'd need to store them in ne_peaks_cwa
    return {wl: 1.0 for wl in filtered_wl}