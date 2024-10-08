import spe_loader


def read_spe(filename):
    """Princeton Instruments spe format"""
    teledyn = spe_loader.load_from_files([filename])
    if len(teledyn.data) != 1 or len(teledyn.data[0]) != 1 or len(teledyn.data[0][0]) != 1:
        raise ValueError('only single spectrum files are supported')
    positions = teledyn.wavelength
    intensities = teledyn.data[0][0][0]
    meta = {}
    meta['@axes'] = ['Wavelengths']
    meta['@signal'] = ''
    return positions, intensities, meta
