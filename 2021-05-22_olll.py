import gmpy2 as gp
import math
import numpy as np
import os

# path_full_in = os.path.join('.', 't1.txt')
# path_full_in = os.path.join('.', 't2.txt')
# path_full_in = os.path.join('.', 't3.txt')
# path_full_in = os.path.join('.', 't4.txt')
# path_full_in = os.path.join('.', 't5.txt')
path_full_in = os.path.join('.', '1606_09320_Basis_of_KL_copy_2021_03_29_emb_80_int.txt')
f_in = open(path_full_in, 'r')
fname = os.path.basename(__file__).replace('.py', '')
path_full_out = path_full_in.replace('.txt', '_{0}.txt'.format(fname))
f_out = open(path_full_out, 'w')

n = int(f_in.readline())
v = []
while True:
    line = f_in.readline()
    if not line:
        break
    line = line.strip().split(', ')
    line[0] = line[0].replace('[', '')
    line[n - 1] = line[n - 1].replace('],', '').replace(']', '')
    # line = [gp.mpfr(i) for i in line]
    line = [int(i) for i in line]
    v.append(line)

import olll

v_init = v.copy()
print(v)
const = 0.75
v = olll.reduction(v, const)
# v = np.array(v)
print(v)

def arr2str_int(v):
    v_str = '['
    for i in range(n):
        v_str += '['
        for j in range(n):
            v_str += '{0}'.format(int(v[i][j]))
            if j < n - 1:   v_str += ', '
            elif i < n - 1: v_str += '],\n'
            else:   v_str += ']]'
    return v_str

def arr2str_float(v):
    v_str = '['
    for i in range(n):
        v_str += '['
        for j in range(n):
            v_str += '{0:.5f}'.format(v[i][j])
            if j < n - 1:   v_str += ', '
            elif i < n - 1: v_str += '],\n'
            else:   v_str += ']]'
    return v_str

def vec2str_float(v):
    v_str = '['
    for i in range(n):
        v_str += '{0:.5f}'.format(v[i])
        if i < n - 1:
            v_str += ', '
    v_str += ']'
    return v_str

vs = v.copy()
def gs(t: int):
    for i in range(t+1):
    # for i in range(n):
        vs[i] = v[i].copy()
        for j in range(i):
            mu = np.dot(v[i], vs[j]) / np.dot(vs[j], vs[j])
            for k in range(n):
                vs[i][k] -= mu * vs[j][k]
    return

gs(n-1)

mus = np.zeros((n, n))
for i in range(n):
    for j in range(i):
        mus[i][j] = np.dot(v[i], vs[j]) / np.dot(vs[j], vs[j])

lovs = np.zeros(n)
for i in range(1, n):
    lovs[i] = np.dot(vs[i], vs[i]) - (const - mus[i][i-1] ** 2) * np.dot(vs[i - 1], vs[i - 1])

print('Check for the LLL reducedness:\n')
print('Mu matrix:\n{0}\n'.format(arr2str_float(mus)))
print('Lovasz vector:\n{0}'.format(vec2str_float(lovs)))
print('Check for the LLL reducedness:\n', file=f_out)
print('Mu matrix:\n{0}\n'.format(arr2str_float(mus)), file=f_out)
print('Lovasz vector:\n{0}'.format(vec2str_float(lovs)), file=f_out)

print(v)
print('Final\n{0}\n'.format(arr2str_int(v)))
print('Final\n{0}\n'.format(arr2str_int(v)), file=f_out)