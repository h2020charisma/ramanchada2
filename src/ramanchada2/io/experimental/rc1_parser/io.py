import os.path
from .third_party_readers import readSPC, readWDF, readOPUS
from .txt_format_readers import read_JCAMP, readTXT
from .binary_readers import readSPA, read_ngs


class UnsupportedFileTypeError(Exception):
    pass


def parse(source_path, file_type=None):
    filename, file_extension = os.path.splitext(source_path)

    if file_type is None:
        file_type = file_extension[1:].lower()  # without the leading dot

    if file_type in {'spc', 'sp'}:
        reader = readSPC
    elif file_type in {'spa'}:
        reader = readSPA
    elif file_type in {'0', '1', '2'}:
        reader = readOPUS
    elif file_type in {'wdf'}:
        reader = readWDF
    elif file_type in {'ngs'}:
        reader = read_ngs
    elif file_type in {'jdx', 'dx'}:
        reader = read_JCAMP
    elif file_type in {'txt', 'txtr', 'csv', 'prn', 'rruf'}:
        reader = readTXT
    else:
        raise UnsupportedFileTypeError(
            f'file type "{file_type}" is not supported'
        )

    x_data, y_data, metadata = reader(source_path)
    if metadata is None:
        metadata = {}
    # Get rid of bytes that are found in some of the formats
    metadata = cleanMeta(metadata)
    # Flatten metadata
    metadata = dict(zip(metadata.keys(), [str(v) for v in metadata.values()]))
    # Extract metadata from native metadata and spectrum data,
    # store in metadata dictionary, and include in CHADA archive.
    metadata["Original file"] = os.path.basename(source_path)
    return x_data, y_data, metadata


def cleanMeta(meta):
    # This cleans complex-strcutures metadata, and returns a dict
    if isinstance(meta, dict):
        meta = {i: meta[i] for i in meta if i != ""}
        for key, value in meta.items():
            meta[key] = cleanMeta(value)
    if isinstance(meta, list):
        for ii, value in enumerate(meta):
            meta[ii] = cleanMeta(value)
    if isinstance(meta, str):
        meta = meta.replace('\\x00', '')
        meta = meta.replace('\x00', '')
    if isinstance(meta, bytes):
        try:
            meta = meta.decode('utf-8')
            meta = cleanMeta(meta)
        except Exception:
            meta = {}
    return meta
