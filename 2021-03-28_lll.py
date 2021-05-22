import gmpy2 as gp
import math
import numpy as np
import os

# path_full_in = os.path.join('.', 't1.txt')
# path_full_in = os.path.join('.', 't2.txt')
# path_full_in = os.path.join('.', 't3.txt')
path_full_in = os.path.join('.', 't4.txt')
# path_full_in = os.path.join('.', 't5.txt')
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
    line = [gp.mpfr(i) for i in line]
    v.append(line)
v = np.array(v)
v_init = v.copy()

def arr2str_int(v):
    v_str = '['
    for i in range(n):
        v_str += '['
        for j in range(n):
            v_str += '{0}'.format(int(v[i, j]))
            if j < n - 1:   v_str += ', '
            elif i < n - 1: v_str += '],\n'
            else:   v_str += ']]'
    return v_str

def arr2str_float(v):
    v_str = '['
    for i in range(n):
        v_str += '['
        for j in range(n):
            v_str += '{0:.5f}'.format(v[i, j])
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

def norm(y):
    return gp.mpfr(math.sqrt(np.dot(y, y)))

print('Initial\n{0}'.format(arr2str_int(v_init)))
print('Initial\n{0}\n'.format(arr2str_int(v_init)), file=f_out)

const = 0.75
# const = 0.99
k = 1
max_k = k
step = 0
cnt_red, cnt_swap = 0, 0
flag_red, flag_swap = False, False

vs = v.copy()
def gs(t: int):
    for i in range(t+1):
    # for i in range(n):
        vs[i] = v[i]
        for j in range(i):
            mu = np.dot(v[i], vs[j]) / np.dot(vs[j], vs[j])
            vs[i] -= mu * vs[j]
    return

gs(n-1)

while k <= n - 1:
    step += 1
    print('Step {0}, k: {1}/{2}'.format(step, k, max_k))
    print('Step {0}, k: {1}/{2}\n'.format(step, k, max_k), file=f_out)

    if flag_swap:
        gs(k)
        flag_swap = False

    for j in range(k-1, -1, -1):
        mu1 = np.dot(v[k], vs[j]) / np.dot(vs[j], vs[j])
        mu1_rnd = round(mu1)
        if mu1_rnd != 0:
            flag_red = True
            v[k] -= np.array(mu1_rnd * v[j], dtype=float) ## Reduction

    if flag_red:
        cnt_red += 1
        gs(k)
        print('red\n{0}\n'.format(arr2str_int(v)))
        print('red\n{0}\n'.format(arr2str_int(v)), file=f_out)

    mu2 = np.dot(v[k], vs[k - 1]) / np.dot(vs[k - 1], vs[k - 1])
    lov = np.dot(vs[k], vs[k]) - (const - mu2 ** 2) * np.dot(vs[k - 1], vs[k - 1])
    if lov >= 0:  ## Lovasz condition
        flag_swap = False
        k += 1
        max_k = k
    else:
        cnt_swap += 1
        flag_swap = True
        v[[k - 1, k]] = v[[k, k - 1]] ## Swap
        k = max(k-1, 1)
        max_k = k

    if flag_swap:
        print('swap\n{0}\n'.format(arr2str_int(v)))
        print('swap\n{0}\n'.format(arr2str_int(v)), file=f_out)

mus = np.zeros((n, n))
for i in range(n):
    for j in range(i):
        mus[i, j] = np.dot(v[i], vs[j]) / np.dot(vs[j], vs[j])

lovs = np.zeros(n)
for i in range(1, n):
    lovs[i] = np.dot(vs[i], vs[i]) - (const - mus[i, i-1] ** 2) * np.dot(vs[i - 1], vs[i - 1])

print('Check for the LLL reducedness:\n')
print('Mu matrix:\n{0}\n'.format(arr2str_float(mus)))
print('Lovasz vector:\n{0}'.format(vec2str_float(lovs)))
print('Check for the LLL reducedness:\n', file=f_out)
print('Mu matrix:\n{0}\n'.format(arr2str_float(mus)), file=f_out)
print('Lovasz vector:\n{0}'.format(vec2str_float(lovs)), file=f_out)

print('Final\n{0}\n'.format(arr2str_int(v)))
print('red: {0}, swap: {1}\n'.format(cnt_red, cnt_swap))
print('Final\n{0}\n'.format(arr2str_int(v)), file=f_out)
print('red: {0}, swap: {1}\n'.format(cnt_red, cnt_swap), file=f_out)