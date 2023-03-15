import numpy as np
import os
import re
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def read_JCAMP(file, verbose=True):
    x = []
    y = []
    # open .txt and read as lines
    with open(file) as d:
        lines = d.readlines()
    end_index = [i for i, s in enumerate(lines) if 'END' in s][-1]
    lines = lines[:end_index]
    # ## Marks meta data
    meta_lines = [ll for ll in lines if ll.startswith('##')][:-1]
    meta = dict([mm.strip('##').strip('\n').split('=') for mm in meta_lines])
    data_lines = [ll for ll in lines if not ll.startswith('##')]
    # read up to second last line
    for ll in data_lines[:-1]:
        # split line into individual numbers
        items = ll.strip('\n').strip().split()
        # 1st is x, the rest is y values
        x.append(float(items[0]))
        [y.append(float(item)) for item in items[1:]]
    # convert to np.array
    # calc num of y per x
    y_per_x = len(y)//len(x)
    x_increment = np.mean(np.diff(np.array(x))) / y_per_x
    new_x = []
    for xx in x:
        for ii in range(y_per_x):
            new_x.append(xx + x_increment*ii)
    # Read last line (may not be complete)
    items = data_lines[-1].strip('\n').strip().split()
    for ii, item in enumerate(items[1:]):
        # 1st is x, the rest is y values
        new_x.append(float(items[0]) + x_increment*ii)
        y.append(float(item))
    return np.array(new_x), np.array(y), meta


def readTXT(file, x_col=0, y_col=0, verbose=True):
    # open .txt and read as lines
    with open(file) as d:
        lines = d.readlines()
    # Find data lines and convert to np.array
    start, stop = startStop(lines)
    logger.debug("Importing " + str(stop-start+1) +
                 " data lines starting from line " + str(start) +
                 " in " + os.path.basename(file) + ".")
    data_lines = lines[start:stop]
    data = dataFromTxtLines(data_lines)
    # if columns not specified, assign x (Raman shift) and y (counts) axes
    if x_col == y_col == 0:
        # x axis is the one with mean closest to 1750
        score = 1./np.abs(data.mean(0)-1750)
        # x axis must be monotonous!
        s = np.sign(np.diff(data, axis=0))
        mono = np.array([np.all(c == c[0]) for c in s.T]) * 1.
        score *= mono
        x_col = np.argmax(score)
        # y axis is the one with maximal std/mean
        score = np.nan_to_num(data.std(0)/data.mean(0), nan=0)
        # Do not choose x axis again for y
        score[x_col] = -1000
        y_col = np.argmax(score)
        # if there's mroe than 2 columns and a header line
        if startStop(lines)[0] > 0 and data.shape[1] > 2:
            logger.debug("Found more than 2 data columns in " +
                         os.path.basename(file) + ".")
            header_line = lines[startStop(lines)[0]-1].strip('\n')
            header_line = [s.casefold() for s in re.split(';|,|\t', header_line)]
            # x axis is header line with "Shift"
            indices = [i for i, s in enumerate(header_line) if 'shift' in s]
            if indices != []:
                x_col = indices[0]
                logger.debug("X data: assigning column labelled '" +
                             header_line[x_col] + "'.")
            else:
                logger.debug("X data: assigning column # " + str(x_col) + ".")
            # y axis is header line with "Subtracted"
            indices = [i for i, s in enumerate(header_line) if 'subtracted' in s]
            if indices != []:
                y_col = indices[0]
                logger.debug("Y data: assigning column labelled '" +
                             header_line[y_col] + "'.")
            else:
                logger.debug("Y data: assigning column # " + str(y_col) + ".")
    x, y = data[:, x_col], data[:, y_col]
    # Only use unique x data points
    x, unique_ind = np.unique(x, return_index=True)
    y = y[unique_ind]
    # is x inverted?
    if all(np.diff(x) <= 0):
        x = np.flip(x)
        y = np.flip(y)
    meta_lines = lines[:startStop(lines)[0]]
    logger.debug("Importing " + str(start-1) + " metadata lines from " +
                 os.path.basename(file) + ".")
    meta_lines = [re.split(';|,|\t|=', ll.strip()) for ll in meta_lines]
    ml = {}
    for ll in meta_lines:
        ml.update({ll[0]: ll[1:]})
    # is x axis pixel numbers instead of Raman shifts?
    if all(np.diff(x) == 1) and (x[0] == 0 or x[0] == 1):
        if "Start WN" in ml:
            start_x = np.int(np.array(ml["Start WN"])[0])
        if "End WN" in ml:
            stop_x = np.int(np.array(ml["End WN"])[0])
        x = np.linspace(start_x, stop_x, len(x))
        logger.debug("X data: using linspace from " + str(start_x) + " to " +
                     str(stop_x) + " 1/cm.")
    return x, y, ml


def dataFromTxtLines(data_lines):
    data = []
    for ii, ll in enumerate(data_lines):
        ll = ll.strip('\n').replace("\t", " ")
        if ";" in ll:
            separator = ";"
        elif "," in ll and "." in ll:
            separator = ","
        elif "," in ll and " " in ll:
            separator = " "
        elif "," in ll:
            separator = ","
        else:
            separator = " "
        items = ll.split(separator)
        items = [item.replace(",", ".") for item in items]
        data.append(items)
    D = pd.DataFrame(np.array(data))
    D = D.replace(r'^\s*$', 0, regex=True)
    D = D.apply(pd.to_numeric)
    D = D.dropna()
    return D.to_numpy()


def isDataLine(line):
    line = line.strip("\n").replace("\t", " ")
    blank = all([c == " " for c in line])
    # has more than 75% digits
    digits = np.sum([d.isdigit() for d in line]) / len(line) > .25
    # apart from digits, has only ".", ";", ",", " "
    chars = all([c in '.,;+-eE ' for c in line if not c.isdigit()])
    return (not blank) & digits & chars


def startStop(lines):
    start_line, stop_line = 0, 0
    for ii, line in enumerate(lines):
        # if this is a data line and the following 5 lines are also data lines, then here is the start line
        if (len(lines) - ii) > 5 and start_line == 0:
            if all([isDataLine(ll) for ll in lines[ii:ii+5]]):
                start_line = ii
        # if this is a data line and the following 5 lines are also data lines, then here is the start line
        if (not isDataLine(line)) and stop_line <= start_line:
            stop_line = ii
    if stop_line <= start_line:
        stop_line = len(lines)-1
    return start_line, stop_line


def getYDataType(y_data):
    types = {0: "Single spectrum", 1: "Line scan", 2: "Map", 3: "Map series / volume"}
    return types[len(y_data.shape)-1]
