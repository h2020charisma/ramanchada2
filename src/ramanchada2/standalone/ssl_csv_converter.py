#!/usr/bin/env python

import argparse
import os

import numpy as np

import ramanchada2 as rc2


def ssl_csv_to_spectra(filename):
    with open(filename) as f:
        lines = [i.strip() for i in f.readlines()]
        assert lines[0].startswith('ssl-SA')
        head_sep = '***** X axis & Intensity *****'
        assert head_sep in lines
        global_meta = {'format': lines[0],
                       'datetime': lines[1],
                       }
        header_sep_idx = lines.index(head_sep)

        global_meta.update(dict([[j.strip() for j in i.split('=')] for i in lines[2:header_sep_idx] if '=' in i]))

        k = 'Laser wavelength'
        if k in global_meta.keys():
            v = global_meta[k]
            if v.endswith(' nm'):
                v = int(v[:-3])
                global_meta['laser wavelength [nm]'] = v

        k = 'Integ time'
        if k in global_meta.keys():
            v = global_meta[k]
            if v.endswith(' ms'):
                v = int(v[:-3])
                global_meta['integration time [ms]'] = v
                global_meta['intigration times(ms)'] = v

        basename = os.path.basename(filename)
        global_meta['Original file'] = basename

        ccc = np.loadtxt(lines[header_sep_idx+1:], delimiter=',')
        ccc = ccc.T
        x = ccc[0]
        spectra = []
        for y_i, y in enumerate(ccc[1:], 1):
            meta = {'spectrum number': y_i, 'total spectra number': ccc.shape[1]}
            meta.update(global_meta)
            spectra.append(rc2.spectrum.Spectrum(x=x, y=y, metadata=meta))
    return spectra


def ssl_csv_converter():
    parser = argparse.ArgumentParser(description='Convert ssl csv file format.')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to ssl csv file')
    parser.add_argument('-t', '--format-to', type=str, required=False,
                        help='Output file format cha or csv', default='cha')
    parser.add_argument('-o', '--output', type=str,
                        help="Output file names in formated string "
                        "Possible options are: {date} {succ_num} {all_spe_num} "
                        "{laser_wavelength} {integration_time} "
                        "{input_basename} {input_extension}"
                        "(default: '{input_basename}--{laser_wavelength}nm"
                        "-{integration_time}ms-{succ_num}-{all_spe_num}.cha')")
    args = parser.parse_args()
    out_fmt = args.format_to or 'cha'
    out_file_format = (
        args.output or
        f'{args.input}-{{laser_wavelength}}nm-{{integration_time}}ms-{{succ_num}}-{{all_spe_num}}.{out_fmt}'
    )

    input_basename, input_extension = os.path.splitext(args.input)

    spes = ssl_csv_to_spectra(args.input)

    for spe_i, spe in enumerate(spes, 1):
        dct = dict(input_basename=input_basename,
                   input_extension=input_extension,
                   date=spe.meta['datetime'].strftime('%F'),
                   succ_num=spe_i,
                   all_spe_num=len(spes),
                   laser_wavelength=spe.meta['laser wavelength [nm]'],
                   integration_time=spe.meta['integration time [ms]'],
                   )

        out_file_name = out_file_format.format(**dct)
        if out_fmt == 'cha':
            spe.write_cha(out_file_name, '/raw')
        elif out_fmt == 'csv':
            spe.write_csv(out_file_name)
        else:
            raise Exception('File format {out_fmt} is unknown')


if __name__ == '__main__':
    ssl_csv_converter()
