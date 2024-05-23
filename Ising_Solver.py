import numpy as np
import time
from utils import read_cnf, Three_sat_toQUBO_v2,getKL, verif_sou
from Ising_algo import runIsingAlgorithm_simulator

def main():

    NITER = 5000
    seed = int(time.time())
    np.random.seed(seed)
    cnf_files = ['sat_k3_v16_c48_s431343124.cnf','sat_k3_v64_c192.cnf']
    cnfs, n, m = read_cnf(cnf_files[1])
    print('n,m', n, m)
    Q, n, m, C = Three_sat_toQUBO_v2(cnfs, n, m)
    MATRIX_SIZE = n + m
    K, L = getKL(Q, MATRIX_SIZE)
    print("start main")
    startTime = time.perf_counter()
    best_candidate= runIsingAlgorithm_simulator(K, L, NITER, MATRIX_SIZE, cnfs, n)
    endTime = time.perf_counter()
    d = endTime - startTime
    print("time for total:")
    print(f"total time: {d}")
    print(f"best energy is: {best_candidate.energy}")
    print(f"best vector is ")
    print(best_candidate.matrix.T)
    S_best = best_candidate.matrix.T[0:n]
    print(S_best)
    num_satclauses = verif_sou(cnfs, S_best, m)
    print('num of sat clauses', num_satclauses)
    print("Done")

if __name__ == "__main__":
    main()
