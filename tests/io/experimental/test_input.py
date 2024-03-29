from ramanchada2.io.experimental.rc1_parser.third_party_readers import readOPUS


def test_opus(opus_experimental_file):
    x, y, meta = readOPUS(opus_experimental_file)
    assert 4096 == len(x)
    assert 4096 == len(y)
    assert 96 == len(meta)
    if "NPT" in meta:
        assert 4096 == meta["NPT"]
    elif "AB Data Parameter.NPT" in meta:
        assert 4096 == meta["AB Data Parameter.NPT"]
