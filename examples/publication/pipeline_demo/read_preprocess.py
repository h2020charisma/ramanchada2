import os.path
import ramanchada2 as rc2

# + tags=["parameters"]
upstream = []
product = None
input_file = None
# -

file = os.path.abspath(input_file)

# load spectrum from file system
spe = rc2.spectrum.from_local_file(file)
spe.plot()

if os.path.isfile(product["cha"]):
    os.remove(product["cha"])

spe.write_cha(product["cha"], dataset="/raw")

spe.subtract_baseline_rc1_snip(niter=40)
spe.write_cha(product["cha"], dataset="/baseline")
