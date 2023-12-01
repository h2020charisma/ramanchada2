
from ramanchada2.io.experimental.rc1_parser.third_party_readers import readOPUS

def test_opus(opus_experimental_file):
    x, y, meta = readOPUS(opus_experimental_file)
    print(x)
    print(meta)
    #assert df.shape == (5, 10), 'wrong shape'