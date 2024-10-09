#!/usr/bin/env python

import argparse
import csv
import os

try:
    from tqdm import trange as range
except ModuleNotFoundError:
    pass

from spectrochempy.core.readers.read_omnic import read_spg


def spg2csv():
    parser = argparse.ArgumentParser(description='Convert SPA/SPG file format.')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to SPA or SPG file')
    parser.add_argument('-o', '--output', type=str,
                        help="Output file names in formated string like "
                        "'/path/to/file-{y_label_name}.csv'. "
                        "Possible options are: {y_label_date} {y_label_name} {y_coordinate} "
                        "{input_basename} {input_extension}"
                        "(default: '{input_basename}-{y_label_name}.csv')")
    args = parser.parse_args()
    out_file_format = args.output or f'{args.input}-{{y_label_name}}.csv'

    input_basename, input_extension = os.path.splitext(args.input)

    spe = read_spg(args.input)

    x = spe.coordset['x'].data
    for i in range(spe.coordset['y'].size):
        (y_label_date, y_label_name), y_coordinate = spe.coordset['y'].labels[i], spe.coordset['y'].data[i]
        y_label_name = str(y_label_name).replace('/', '_')
        out_file_name = out_file_format.format(basename=input_basename,
                                               y_label_date=y_label_date,
                                               y_label_name=y_label_name,
                                               y_coordinate=y_coordinate)
        y = spe.data[i]
        print(out_file_name)
        with open(out_file_name, 'w') as fp:
            csv_writer = csv.writer(fp)
            csv_writer.writerows(zip(x, y))


if __name__ == '__main__':
    spg2csv()
