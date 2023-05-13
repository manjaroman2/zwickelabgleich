import numpy as np 
import matplotlib.pyplot as plt 
import argparse
from pathlib import Path
from common import readcsv, dir_path, get_parser, ON_COLAB
from fehlergeraden import calc as fehlergeraden_calc


PNG_DPI = 900

def calc(X, Y, DX, DY, xquantity, xunit, yquantity, yunit, plot=True, png=False, out=Path.cwd(), debug=False, download=False, to_download=list(), fehlergeraden=False, *args):    
    # print(f'{xquantity} [{xunit}]  | ', f'{yquantity} [{yunit}]')
    # print(DX, DY)
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
    
    # print(X[_max:], Y[_max:], DX[_max:], DY[_max:])
    if fehlergeraden:
        _maxx = 5 
        upper_h, upper_l = fehlergeraden_calc(X[_maxx:], Y[_maxx:], DX[_maxx:], DY[_maxx:])
        print(upper_h, upper_l)
        lower_h, lower_l = fehlergeraden_calc(X[:_min], Y[:_min], DX[:_min], DY[:_min])
    
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
    if png or plot:
        plt.close()
        plt.cla()
        plt.clf()
        plt.xlabel(f"{xquantity} [{xunit}]")
        plt.ylabel(f"{yquantity} [{yunit}]")
        plt.minorticks_on()
        plt.grid(True, which='both', axis='both')
        plt.vlines(X[i], R_lower, R_upper, colors='c', linestyles='solid')
        plt.plot([0, X[-1]], [upper_t, Y[-1]], c='r', linestyle='dashed')
        plt.plot([0, X[-1]], [Y[0], lower_m*X[-1]+lower_t], c='g', linestyle='dashed')
        if fehlergeraden:
            if upper_h:
                plt.plot([upper_h[0][0], upper_h[1][0]], [upper_h[1][0], upper_h[1][1]], c='r', linestyle='dashed')
        # plt.plot(X, Y, marker='x')
        plt.scatter(X, Y, marker='x')
        
        if len(DX) > 0 and len(DY) > 0:
            plt.errorbar(X, Y, xerr=DX, yerr=DY, fmt='_', ecolor="black", solid_capstyle='projecting', capsize=5)
        elif len(DX) > 0 and not  len(DY) > 0:
            plt.errorbar(X, Y, xerr=DX, yerr=np.zeros_like(DX), fmt='_', ecolor="black", solid_capstyle='projecting', capsize=5)
            
        plt.text(0, R_upper * 0.995, f'{round(R_upper, 2)}{yunit}', ha='left', va='center')
        plt.text(0, R_lower * 1.005, f'{round(R_lower, 2)}{yunit}', ha='left', va='center')
        plt.hlines([R_lower, R_upper], 0, X[i], colors='orange', linestyles='solid')
        if png:
            # {xquantity}[{xunit}]_{yquantity}[{yunit}]
            fn = out / Path(f"{file.stem}_zwickelabgleich.png")
            plt.savefig(fn, dpi=PNG_DPI)
            print(f"-> {fn.resolve()}")
            if ON_COLAB and download:
                to_download.append(fn)
        if plot:
            plt.show()
    return R_lower, R_upper

def save_to_xlsx():
    import openpyxl
    wb = openpyxl.load_workbook('input.xlsx')
    ws = wb.active
    img = openpyxl.drawing.image.Image('myplot.png')
    img.anchor(ws.cell('A1'))

    ws.add_image(img)
    wb.save('output.xlsx')



if __name__ == "__main__":    
    parser = get_parser()
    parser.add_argument('--out', type=dir_path, default=Path.cwd())
    parser.add_argument('files', help="path to csv files", type=argparse.FileType('r'), nargs='+')
    parser.add_argument('-pl', '--plot', action='store_true', help="plot with matplotlib")
    parser.add_argument('-png', '--png', action='store_true', help="save as png")
    parser.add_argument('-f', '--fehlergeraden', action='store_true', help="calculate fehlergeraden")
    args = parser.parse_args()

    if args.download and not ON_COLAB:
        print(f"Warning: --download flag was supplied but 'google' module was not found, we are either not on colab or something went wrong")
    # print('  '.join([f"{k}={w}"for k, w in args.__dict__.items()]))

    to_download = []
    for f in args.files:
        file = Path(f.name).resolve()
        calc(*readcsv(file), png=args.png, plot=args.plot, debug=args.debug, download=args.download, out=args.out, to_download=to_download, fehlergeraden=args.fehlergeraden)

    for f in to_download:
        colab_files.download(f)