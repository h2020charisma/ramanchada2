from ramanchada2.io.experimental import rc1_parser


def test_opus(opus_experimental_file):
    x, y, meta = rc1_parser.parse(opus_experimental_file, filetype='0')
    print(x)
    print(meta)
    #assert df.shape == (5, 10), 'wrong shape'