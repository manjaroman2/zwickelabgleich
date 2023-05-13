import numpy as np 
import matplotlib.pyplot as plt 
from pathlib import Path
from common import readcsv, dir_path, slope, get_parser, get_parser_type_file


# https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


def calc(X, Y, DX, DY, *args, R=200, plot=False, debug=False):
    assert len(X) == len(Y)
    print(f"data={len(X)}  R={R}")
    M, T = np.linalg.lstsq(np.vstack([X, np.ones(len(X))]).T, Y, rcond=None)[0]
    diagonal_H = []
    diagonal_L = []
    if M > 0: 
        for i in range(len(X)):
            diagonal_H.append(np.array([[X[i] + DX[i], Y[i] - DY[i]], [X[i] - DX[i], Y[i] + DY[i]]]))
            diagonal_L.append(np.array([[X[i] - DX[i], Y[i] - DY[i]], [X[i] + DX[i], Y[i] + DY[i]]]))
    else:
        for i in range(len(X)):
            diagonal_H.append(np.array([[X[i] - DX[i], Y[i] - DY[i]], [X[i] + DX[i], Y[i] + DY[i]]]))
            diagonal_L.append(np.array([[X[i] + DX[i], Y[i] - DY[i]], [X[i] - DX[i], Y[i] + DY[i]]]))
    
    highest_M = -np.Infinity
    lowest_M = np.Infinity
    highest_T = np.nan
    lowest_T = np.nan
    h = None
    l = None
    
    d = diagonal_H[0]
    dSF_H = d[0]
    dS = d[1]-d[0]
    LS = np.linalg.norm(dS)
    dSn_H = dS/LS
    
    d = diagonal_H[-1]
    dEF_H = d[1]
    dE = d[0]-d[1]
    LE = np.linalg.norm(dE)
    dEn_H = dE/LE
    
    d = diagonal_L[0]
    dSF_L = d[1]
    dSn_L = (d[0]-d[1])/LS
    
    d = diagonal_L[-1]
    dEF_L = d[0]
    dEn_L = (d[1]-d[0])/LE
    
    s_range = np.linspace(0, LS, num=R, endpoint=True)
    e_range = np.linspace(0, LE, num=R, endpoint=True)
    for s in s_range: 
        if h and l: break
        d_s_H = dSF_H+dSn_H*s
        d_s_L = dSF_L+dSn_L*s
        # plt.scatter(*d_s_H, color="c")
        # plt.scatter(*d_s_L, color="c")
        for e in e_range:
            d_e_H = dEF_H+dEn_H*e
            # d_s_H -> d_e_H 
            # if debug:
            #     plt.plot([d_s_H[0], d_e_H[0]], [d_s_H[1], d_e_H[1]], color="red", alpha=0.2, linestyle='-.')
            for c in diagonal_H[1:-1]: 
                # c[0] -> c[1]
                if not intersect(d_s_H, d_e_H, c[0], c[1]):
                    break
            else:
                m = slope(d_s_H, d_e_H)
                if m > highest_M:
                    highest_M = m
                    h = [d_s_H, d_e_H]
            d_e_L = dEF_L+dEn_L*e
            # d_s_L -> d_e_L 
            if debug:
                plt.plot([d_s_L[0], d_e_L[0]], [d_s_L[1], d_e_L[1]], color="green", alpha=0.2, linestyle='-.')
            # c[0] -> c[1]
            for c in diagonal_L[1:-1]: 
                if not intersect(d_s_L, d_e_L, c[0], c[1]): break
            else:
                m = slope(d_s_L, d_e_L)
                if m < lowest_M: 
                    lowest_M = m
                    l = [d_s_L, d_e_L]
    if h: 
        highest_T = h[0][1] - highest_M*h[0][0]
        h = [[X[0], highest_M*X[0]+highest_T], [X[-1], highest_M*X[-1]+highest_T]]
        print(h)
    else:
        print("No highest slope found!")
        # raise Exception("No highest slope found!")
    if l: 
        lowest_T = l[0][1] - lowest_M*l[0][0]
        l = [[X[0], lowest_M*X[0]+lowest_T], [X[-1], lowest_M*X[-1]+lowest_T]]
        print(l)
    else:
        print("No lowest slope found!")
        # raise Exception("No lowest slope found!")
    
    if plot:    
        plt.minorticks_on()
        plt.grid(True, which='both', axis='both')
        plt.scatter(X, Y, marker='x')
        
        """ plot diagonals """ 
        if debug: 
            for d in diagonal_H:
                plt.plot([d[0][0], d[1][0]], [d[0][1], d[1][1]], marker='', color='magenta')
            for d in diagonal_L:
                plt.plot([d[0][0], d[1][0]], [d[0][1], d[1][1]], marker='', color='blue')
        # plt.plot([dSF[0], dS[0]+dSF[0]], [dSF[1], dS[1]+dSF[1]], marker='', color='red')

        if h: 
            plt.plot([h[0][0], h[1][0]], [h[0][1], h[1][1]], color='red', marker='')
        if l: 
            plt.plot([l[0][0], l[1][0]], [l[0][1], l[1][1]], color='green', marker='')
        
        plt.plot([X[0]-DX[0], X[-1]+DX[-1]], [M*(X[0]-DX[0])+T, M*(X[-1]+DX[-1])+T], color='goldenrod', marker='', linestyle='--')
        if len(DX) > 0 and len(DY) > 0:
            plt.errorbar(X, Y, xerr=DX, yerr=DY, fmt='_', ecolor="black")
        elif len(DX) > 0 and not len(DY) > 0:
            plt.errorbar(X, Y, xerr=DX, yerr=np.zeros_like(DX), fmt='_', ecolor="black")
        plt.show()
    return h, l


if __name__ == "__main__":    
    parser = get_parser()
    parser.add_argument('--out', type=dir_path, default=Path.cwd())
    parser.add_argument('files', help="path to csv files", type=get_parser_type_file(), nargs='+')
    parser.add_argument('-pl', '--plot', action='store_true', help="plot with matplotlib")
    parser.add_argument('-R', type=int, default=200, help="resolution")
    args = parser.parse_args()
    for f in args.files:
        file = Path(f.name).resolve()
        h, l = calc(*readcsv(file),  R=args.R, plot=args.plot, debug=args.debug)
        print(h)
        print(l)
