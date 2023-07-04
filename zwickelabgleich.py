import argparse
import os 
from pathlib import Path
import numpy as np 
import sys 

import locale
# locale.setlocale(locale.LC_ALL, "de_DE.UTF-8")
locale.setlocale(locale.LC_NUMERIC, "de_DE.UTF-8")
this_module = sys.modules[__name__]

ndecimals = 2
error = 0.03

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)
    
parser = argparse.ArgumentParser()
parser.add_argument('-debug', action='store_true', help="debug mode")
parser.add_argument('--out', type=dir_path, default=Path.cwd())
parser.add_argument('files', help="path to csv files", type=argparse.FileType('r'), nargs='+')
parser.add_argument('-pl', '--plot', action='store_true', help="plot with matplotlib")
parser.add_argument('-png', '--png', action='store_true', help="save as png")
parser.add_argument('-e', '--err', action='store_true', help="calculate the error")
args = parser.parse_args()



def readcsv(file: Path):
    data = np.genfromtxt(file, dtype=float, delimiter=',', names=True, missing_values="nan", filling_values=np.nan)
    cols = {"t": ["s", "min"], "T1": ["°C", "K"], "T2": ["°C", "K"], "TM": ["°C", "K"]}
    datacols = []
    for x in data.dtype.names:
        x_sp = x.split("_")
        if len(x_sp) != 2:
            raise Exception(f"Column needs to be formatted like '<t,T1,T2,TM> <unit>\n' {x}")
        if x_sp[0] not in cols: 
            raise Exception(f"Unknown column {x}")
        if x_sp[1] not in cols[x_sp[0]]: 
            raise Exception(f"Unknown unit {x} (has to be {cols[x_sp[0]]})")
        datacols.append(x_sp)

    # print("valid columns:", datacols)

    dataincols = list(zip(*data))
    for i in range(len(datacols)):
        setattr(this_module, datacols[i][0], dataincols[i])
    
    t = getattr(this_module, "t")
    T1 = getattr(this_module, "T1")
    T2 = getattr(this_module, "T2")
    TM = getattr(this_module, "TM")
    
    return t, T1, T2, TM



def calc(t, T1, T2, TM, png=False, plot=False, debug=False, out=Path.cwd(), err=False):
    t = np.array(t)
    T1 = np.array(T1)
    T2 = np.array(T2)
    TM = np.array(TM)
    t1len = np.argmax(T1)
    
    idx = np.isfinite(T1)
    M1, B1 = np.polyfit(t[idx], T1[idx], 1)
    G1 = lambda x: M1*x + B1
    idx = np.isfinite(T2)
    M2, B2 = np.polyfit(t[idx], T2[idx], 1)
    G2 = lambda x: M2*x + B2
    
    TM_notnan = TM[t1len:]
    t_TM = t[t1len:]
    index_max = np.argmax(TM_notnan)
    MM, BM = np.polyfit(t_TM[index_max:], TM_notnan[index_max:], 1)
    GM = lambda x: MM*x + BM
    
    if plot: 
        lines = [[], []]

    
    def big_ahh_formula(x1, x2, y1, y2, d):
        t1 = B1 
        t2 = B2 
        m1 = M1 
        m2 = M2
        tM = BM
        mM = MM 
        xM = (-mM*x1 + mM*x2 - 2*t1 + 2*t2 + y1 - y2 - np.sqrt(-8*d*m1 + 8*d*m2 + 4*m1**2*x1**2 - 4*m1*m2*x1**2 - 4*m1*m2*x2**2 + 8*m1*t1*x1 - 8*m1*t2*x2 - 4*m1*tM*x1 + 4*m1*tM*x2 - 4*m1*x1*y1 + 4*m1*x2*y2 + 4*m2**2*x2**2 - 8*m2*t1*x1 + 8*m2*t2*x2 + 4*m2*tM*x1 - 4*m2*tM*x2 + 4*m2*x1*y1 - 4*m2*x2*y2 + mM**2*x1**2 - 2*mM**2*x1*x2 + mM**2*x2**2 + 4*mM*t1*x1 - 4*mM*t1*x2 - 4*mM*t2*x1 + 4*mM*t2*x2 - 2*mM*x1*y1 + 2*mM*x1*y2 + 2*mM*x2*y1 - 2*mM*x2*y2 + 4*t1**2 - 8*t1*t2 - 4*t1*y1 + 4*t1*y2 + 4*t2**2 + 4*t2*y1 - 4*t2*y2 + y1**2 - 2*y1*y2 + y2**2))/(2*(m1 - m2))
        return xM
        
    def dks():
        lower_sum = 0
        for k in range(index_max + 1):
            lower_sum += TM_notnan[k] - G1(t_TM[k])
            upper_sum = sum([GM(t_TM[j]) - TM_notnan[j] for j in range(k, index_max + 1)])
            if lower_sum > upper_sum:
                break
            if lower_sum == upper_sum: 
                return t_TM[k]

        x_1 = t_TM[k - 1] 
        y_1 = TM_notnan[k - 1]
        x_2 = t_TM[k] 
        y_2 = TM_notnan[k]
        d = upper_sum - lower_sum
        if plot:
            lines[0].append(x_1)
            lines[0].append(x_2)
            lines[1].append(y_1)
            lines[1].append(y_2)
        return big_ahh_formula(x_1, x_2, y_1, y_2, d)
        
           
    xM = dks()
    T_1 = G1(xM)
    T_M = GM(xM)
    T_2 = G2(xM)
    
    xM_1 = xM*(1-error)
    xM_2 = xM*(1+error)
    delta_T_1 = abs(G1(xM_2)-G1(xM_1))/2
    delta_T_M = abs(GM(xM_2)-GM(xM_1))/2
    delta_T_2 = abs(G2(xM_2)-G2(xM_1))/2
    
    
    import matplotlib.pyplot as plt
    plt.scatter(t, T1, label="T1", marker='x')
    plt.scatter(t, T2, c='r', label="T2", marker='x')
    plt.scatter(t, TM, label="TM", marker='x')
    plt.plot([t[0], t[-1]], [G1(t[0]), G1(t[-1])])
    plt.plot([t[0], t[-1]], [G2(t[0]), G2(t[-1])], c='r')
    plt.plot([t[0], t[-1]], [GM(t[0]), GM(t[-1])])
    plt.vlines(x=[xM, xM_1, xM_2], ymin=[T_1, T_1-delta_T_1, T_1+delta_T_1], ymax=[T_2, T_2+delta_T_2, T_2-delta_T_2], colors=['purple', 'plum', 'plum'], linestyles=['solid', 'dashed', 'dashed'])
    plt.hlines(y=[T_1, T_M, T_2], xmin=[0], xmax=[xM], linestyles=[":"], colors=['purple'])
    props = dict(boxstyle='round', facecolor='wheat', alpha=1)
    plt.text(0, T_1, f'T₁={round(T_1, ndecimals):.2f} °C'.replace(".", ","), ha='left', va='center', bbox=props)
    plt.text(0, T_M, f'Tₘ={round(T_M, ndecimals):.2f} °C'.replace(".", ","), ha='left', va='center', bbox=props)
    plt.text(0, T_2, f'T₂={round(T_2, ndecimals):.2f} °C'.replace(".", ","), ha='left', va='center', bbox=props)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(xM_2, T_2+delta_T_2, f'ΔT₂=±{round(delta_T_2, ndecimals):.2f} °C'.replace(".", ","), ha='left', va='bottom', bbox=props)
    plt.text(xM_2, T_M, f'ΔTₘ=±{round(delta_T_M, ndecimals):.2f} °C'.replace(".", ","), ha='left', va='bottom', bbox=props)
    plt.text(xM_2, T_1-delta_T_1, f'ΔT₁=±{round(delta_T_1, ndecimals):.2f} °C'.replace(".", ","), ha='left', va='bottom', bbox=props)
    plt.grid(color='gray', linestyle='-', which='major', linewidth=0.5)
    plt.grid(color='lightgray', linestyle='-', which='minor', linewidth=0.5)
    plt.minorticks_on()
    plt.xlabel('t / [s]')
    plt.ylabel('T / [°C]')
    if plot:
        plt.show()
        
    print(f"T₁={round(T_1, 2)} °C")
    print(f'Tₘ={round(T_M, 2)} °C')
    print(f'T₂={round(T_2, 2)} °C')
    
    if png:
        png: Path = png
        print(f">>> {file.stem}.png")
        plt.savefig(f'{file.stem}.png', bbox_inches='tight')
    
        
    
for f in args.files:
    file = Path(f.name).resolve()
    png = False
    if args.png:
        png = file
    calc(*readcsv(file), png=png, plot=args.plot, debug=args.debug, out=args.out, err=args.err)
