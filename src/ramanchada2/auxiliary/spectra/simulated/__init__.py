import importlib.resources

loc = importlib.resources.files(__name__)


FILES = {
    'calcite_crystal_raw': './crystal/calcite_CRYSTAL_PBE_raw_data.dat',
    'calcite_crystal_convolved': './crystal/calcite_CRYSTAL_PBE_convoluted.dat',
    'calcite_vasp': './vasp/snCAL_vasp_raman_ALL.dat',
}


for f in FILES:
    FILES[f] = str(loc.joinpath(FILES[f]))
