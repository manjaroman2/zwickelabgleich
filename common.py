import numpy as np 
from pathlib import Path
import os 
import argparse

import importlib 
if (ON_COLAB := importlib.util.find_spec("google")):
    from google.colab import files as colab_files

def slope(A, B):
    return (B[1]-A[1])/(B[0]-A[0])


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)
    
def readcsv(file: Path):
    data = np.genfromtxt(file, dtype=None, delimiter=',', names=True)
    xcol = data.dtype.names[0]
    ycol = data.dtype.names[1]
    hasxerror = len(data.dtype.names) > 2
    hasyerror = len(data.dtype.names) > 3
    DX = np.array([])
    DY = np.array([])
    if hasxerror: 
        xdelta = data.dtype.names[2]
        DX = data[xdelta]
        
    if hasyerror:
        ydelta = data.dtype.names[3]
        DY = data[ydelta]
    
    if hasxerror or hasyerror:
        if not hasyerror:
            for i in range(len(DX)):
                if np.isnan(DX[i]):
                    DX[i] = DX[i-1]
        else:
            for i in range(len(DX)):
                if np.isnan(DX[i]):
                    DX[i] = DX[i-1]
                if np.isnan(DY[i]):
                    DY[i] = DY[i-1]
    xquantity, xunit = xcol.replace("_", " ").split()
    yquantity, yunit = ycol.replace("_", " ").split()
    # print(f'{xquantity} [{xunit}]  | ', f'{yquantity} [{yunit}]')
    X = data[xcol]
    Y = data[ycol]
    
    return X, Y, DX, DY, xquantity, xunit, yquantity, yunit

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dwn', '--download', action='store_true', help="download when running on google colab")
    parser.add_argument('-debug', action='store_true', help="debug mode")

    return parser

def get_parser_type_file():
    return argparse.FileType('r')