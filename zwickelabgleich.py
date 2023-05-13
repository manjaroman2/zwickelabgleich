import numpy as np 
import argparse
from pathlib import Path
import os

PNG_DPI = 600

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=dir_path, default=Path.cwd())
parser.add_argument('files', help="path to csv files", type=argparse.FileType('r'), nargs='+')
parser.add_argument('-pl', '--plot', action='store_true', help="plot with matplotlib")
parser.add_argument('-png', '--png', action='store_true', help="save as png")
args = parser.parse_args()

# print('  '.join([f"{k}={w}"for k, w in args.__dict__.items()]))

def calc(file: Path, data: np.array):
    xcol, ycol = data.dtype.names
    xquantity, xunit = xcol.replace("_", " ").split()
    yquantity, yunit = ycol.replace("_", " ").split()
    # print(f'{xquantity} [{xunit}]  | ', f'{yquantity} [{yunit}]')
    X = data[xcol]
    Y = data[ycol]
    _max = len(Y) - np.argmax(Y[::-1]) - 1
    xmax = X[_max]
    ymax = Y[_max]
    upper_m = (ymax - Y[-1])/(xmax - X[-1])
    upper_t = ymax - upper_m * xmax 
    _min = len(Y) - np.argmin(Y[::-1]) - 1
    xmin = X[_min]
    ymin = Y[_min]
    if (_d := (xmin - X[0])) == 0:
        _d = 1
    lower_m = (ymin - Y[0])/_d
    lower_t = ymin - lower_m * xmin 
    
    i = _min
    last_diff = 0
    while True: 
        lowersum = sum([Y[j] - (lower_m * X[j] + lower_t) for j in range(_min, i)])
        uppersum = sum([(upper_m * X[j] + upper_t) - Y[j] for j in range(i, _max)])
        
        diff = uppersum - lowersum
        # print(diff)
        if diff > 0:
            i += 1
            last_diff = diff
            continue
        elif abs(diff) > last_diff:
            i -= 1
            break
        else:
            break
    R_upper = upper_m*X[i]+upper_t
    R_lower = lower_m*X[i]+lower_t
    print(f"{file.stem}:")
    print(f"lower={round(R_lower, 1)}")
    print(f"upper={round(R_upper, 1)}")
    if args.png or args.plot:
        import matplotlib.pyplot as plt 
        plt.close()
        plt.xlabel(f"{xquantity} [{xunit}]")
        plt.ylabel(f"{yquantity} [{yunit}]")
        plt.minorticks_on()
        plt.grid(True, which='both', axis='both')
        plt.vlines(X[i], R_lower, R_upper, colors='k', linestyles='solid')
        plt.plot([0, X[-1]], [upper_t, Y[-1]], c='r', linestyle='dashed')
        plt.plot([0, X[-1]], [Y[0], lower_m*X[-1]+lower_t], c='g', linestyle='dashed')
        # plt.plot(X, Y, marker='x')
        plt.scatter(X, Y, marker='x')
        plt.hlines([R_lower, R_upper], 0, X[i], colors='orange', linestyles='solid')
        plt.text(0, R_upper * 0.995, f'{round(R_upper, 2)}{yunit}', ha='left', va='center')
        plt.text(0, R_lower * 1.005, f'{round(R_lower, 2)}{yunit}', ha='left', va='center')
        if args.png:
            # {xquantity}[{xunit}]_{yquantity}[{yunit}]
            fn = args.out / Path(f"{file.stem}_zwickelabgleich.png")
            plt.savefig(fn, dpi=PNG_DPI)
            print(f"-> {fn.resolve()}")
        if args.plot:
            plt.show()
        plt.close()
    return R_lower, R_upper


for f in args.files:
    file = Path(f.name).resolve()
    # print(file)
    R_lower, R_upper = calc(file, np.genfromtxt(file, dtype=None, delimiter=',', names=True))
