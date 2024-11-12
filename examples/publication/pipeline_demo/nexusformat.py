import ramanchada2 as rc2
import pyambit.datamodel as mx
from pyambit.nexus_spectra import spe2ambit
import nexusformat.nexus.tree as nx
import uuid
import os.path

# + tags=["parameters"]
upstream = ["read_preprocess"]
product = None
sample = None
laser_wl = None
instrument_make = None
instrument_model = None
data_provider = None
# -

spe_raw = rc2.spectrum.from_chada(upstream["read_preprocess"]["cha"], dataset="/raw")
spe_raw.plot()

prefix = "DEMO"

substances = []
substance = mx.SubstanceRecord(name=sample, publicname=sample, ownerName="TEST")
substance.i5uuid = "{}-{}".format(prefix, uuid.uuid5(uuid.NAMESPACE_OID, sample))
protocol_application = spe2ambit(
    x=spe_raw.x,
    y=spe_raw.y,
    meta=spe_raw.meta,
    instrument=(instrument_make, instrument_model),
    wavelength=laser_wl,
    provider=data_provider,
    investigation="My experiment",
    sample=sample,
    sample_provider="CHARISMA",
    prefix=prefix,
    endpointtype="RAW_DATA",
    unit="cm-1",
)
substance.study = [protocol_application]
substances.append(substance)

output_file = product["nxs"]

substances = mx.Substances(substance=substances)
nxroot = nx.NXroot()
nxroot.attrs["pyambit"] = "0.0.1"
nxroot.attrs["file_name"] = os.path.basename(output_file)
substances.to_nexus(nxroot)

print(nxroot.tree)
nxroot.save(output_file, mode="w")
