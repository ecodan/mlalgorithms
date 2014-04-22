__author__ = 'dan'

'''
Simple (and slow) implementation of the Transfer Entropy (TE) and Generalized Transfer Entropy (GTE)
that calculates a measure of causality between parallel event streams.
'''
import pandas as pd
import numpy as np
import math
import time

# returns avg time, time remaining
def time_remaining(tot_iters, cur_iter, total_dur):
    avg = total_dur/(cur_iter*1.0)
    rem = (tot_iters - cur_iter) * avg
    return avg/1000, rem/1000


# input is a numpy MxN matrix of binary ints (0, 1)
def TE(matrix):
    print('starting TE, matrix shape=' + str(matrix.shape))
    M = matrix.shape[0] # lentgh
    N = matrix.shape[1] # width

    # create a return structure matrix NxN
    ret = np.array(np.zeros((N,N)))

    start = time.time() * 1000
    n_start = 0.0
    n_end = 0.0
    iters = 0
    # loop through all pairwise combination
    for i in range(0,N):
        for j in range(0,N):
            if i == j: continue
            iters += 1
            print('processsing ' + str(i) + ',' + str(j))

            n_start = time.time() * 1000

            xn1xnyn = np.zeros((2,2,2))
            xn1xn = np.zeros((2,2))
            xnyn = np.zeros((2,2))

            p1 = (1.0*np.sum(matrix[::,i]))/M
            pxn = [1-p1, p1]

            # loop through each time frame and compute the current equation components
            for m in range(1,M):
                # get the values of the three key component; assumes values are 0 or 1
                xn = matrix[m-1,i]
                yn = matrix[m-1,j]
                xn1 = matrix[m,i]

                # increment the counts in the appropriate "bins"
                xn1xnyn[xn1,xn,yn] += 1
                xn1xn[xn1,xn] += 1
                xnyn[xn,yn] += 1

            # calculate probablilities of all combos
            pxn1xnyn = xn1xnyn/(M-1)
            pxn1xn = xn1xn/(M-1)
            pxnyn = xnyn/(M)

            # print('pxn1xnyn=\n' + str(pxn1xnyn))
            # print('pxn1xn=\n' + str(pxn1xn))
            # print('pxnyn=\n' + str(pxnyn))
            # print('px1=' + str(pxn[1]))
            # print('px0=' + str(pxn[0]))

            # calculate transfer entropy
            aTij = []
            for _xn1 in [0,1]:
                for _xn in [0,1]:
                    for _yn in [0,1]:
                        lv = ((pxn1xnyn[_xn1, _xn, _yn] * pxn[_xn]) / (pxn1xn[_xn1, _xn] * pxnyn[_xn, _yn]))
                        T = pxn1xnyn[_xn1, _xn, _yn] * np.log10(lv)
                        # print(str(_xn1) + ',' + str(_xn) + ',' + str(_yn) + ': ' + str(lv))
                        aTij.append(T)

            Tij = np.nansum(aTij)
            n_end = time.time() * 1000
            print('Tij for ' + str(i) + '->' + str(j) + ' = ' + str(Tij) + ' | dur=' + str(n_end-n_start) + ' | avg/rem=' + str(time_remaining(N**2, iters, n_end-start)))
            ret[i][j] = Tij


    return ret


def array_to_num_str(a):
    ret = ''
    for i in range(0, len(a)):
        ret += str(0 if a[i] == 0 else 1)
    return ret

def array_to_int(a):
    ret = 0
    for i in range(0, len(a)):
        exp = len(a)-1-i
        ret += 2**exp if a[i] != 0 else 0
    return ret


# input is a numpy MxN matrix of binary ints (0, 1)
def GTE(matrix, k):
    print('starting GTE, matrix shape=' + str(matrix.shape))
    M = matrix.shape[0] # lentgh
    N = matrix.shape[1] # width

    # create a return structure matrix NxN
    ret = np.array(np.zeros((N,N)))

    start = time.time() * 1000
    n_start = 0.0
    n_end = 0.0
    iters = 0

    # loop through all pairwise combination
    for i in range(0,N):
        for j in range(0,N):
            if i == j: continue
            iters += 1
            print('processsing ' + str(i) + ',' + str(j))

            n_start = time.time() * 1000

            xn1xnyn1 = np.zeros((2,2**k,2**k))
            xn1xn = np.zeros((2,2**k))
            xnyn1 = np.zeros((2**k,2**k))

            p1 = (1.0*np.sum(matrix[::,i]))/M
            pxn = [1-p1, p1]

            # loop through each time frame and compute the current equation components
            for m in range(k+1,M):
                # get the values of the three key component; assumes values are 0 or 1
                xn = matrix[m-k-1:m-1,i]
                yn1 = matrix[m-k:m,j]
                xn1 = 0 if matrix[m,i] == 0 else 1

                # increment the counts in the appropriate "bins"
                # print('diag ' + str(xn1) + '|' + array_to_num_str(xn) + '|' + array_to_num_str(yn1))
                vxn = array_to_int(xn)
                vyn1 = array_to_int(yn1)

                xn1xnyn1[xn1, vxn, vyn1] += 1
                xn1xn[xn1, vxn] += 1
                xnyn1[vxn, vyn1] += 1

            # calculate probablilities of all combos
            pxn1xnyn1 = xn1xnyn1/(M-1)
            pxn1xn = xn1xn/(M-1)
            pxnyn1 = xnyn1/(M)

            # print('pxn1xnyn=\n' + str(pxn1xnyn))
            # print('pxn1xn=\n' + str(pxn1xn))
            # print('pxnyn=\n' + str(pxnyn))
            # print('px1=' + str(pxn[1]))
            # print('px0=' + str(pxn[0]))

            # calculate transfer entropy
            aTij = []
            for _xn1 in [0,1]:
                for _xn in range(0,2**k):
                    for _yn1 in range(0,2**k):
                        lv = ((pxn1xnyn1[_xn1, _xn, _yn1] * pxn[_xn1]) / (pxn1xn[_xn1, _xn] * pxnyn1[_xn, _yn1]))
                        T = pxn1xnyn1[_xn1, _xn, _yn1] * np.log10(lv)
                        aTij.append(T)

            Tij = np.nansum(aTij)
            n_end = time.time() * 1000
            print('Tij for ' + str(i) + '->' + str(j) + ' = ' + str(Tij) + ' | dur=' + str(n_end-n_start) + ' | avg/rem=' + str(time_remaining(N**2, iters, n_end-start)))
            ret[i][j] = Tij
    return ret


def test():
    matrix = np.array([
        [1,1],
        [1,0],
        [1,1],
        [0,0],
        [1,1],
        [1,0],
        [0,1],
        [1,0],
        [0,1],
        [0,1],
        [1,0],
        [1,1],
        [0,1],
        [0,0],
        [1,1],
        [1,1],
        [1,0],
        [0,1],
        [1,1],
        [1,0],
        [0,0],
        [1,1]])

    ret = TE(matrix)
    print('TE ret = ' + str(ret))

    ret = GTE(matrix, 3)
    print('GTE ret = ' + str(ret))
