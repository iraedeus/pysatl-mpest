from em import GaussianModel, WeibullModel, EM
import numpy as np
from itertools import combinations
from scipy.stats import norm, weibull_min
import sys
import time
import types


def test(data, count):
    model, x, start = data

    start = [(np.log(k), np.log(l)) for (k, l) in start]

    result = []
    for _ in range(count):
        em = EM(model(), 0.01, 75)
        tic = time.perf_counter()
        em.fit(x, np.array(start), len(start))
        toc = time.perf_counter()
        emo = [list(t) for t in em.o]
        emw = list(em.w)
        result.append((emo, emw, em.step, toc - tic))
        if em.step >= 75:
            break

    return result


def run_test(p, start, count):
    start = int(start)
    count = int(count)
    with open(p, "r") as f:
        file = f.readlines()
    with open(p + f"result_{start}.txt", "w") as out:
        for l in file[start::count]:
            splitted = l.split(";")
            ind, distr_count = (int(i) for i in splitted[:2])
            print(ind)
            O = []
            for i in range(distr_count):
                O.append(tuple((float(t) for t in splitted[2 + i * 2 : 4 + i * 2])))

            start_O = []
            for i in range(distr_count):
                r = 2 + distr_count * 2 + i * 2
                start_O.append(tuple((float(t) for t in splitted[r : r + 2])))
            x = [float(t) for t in splitted[4 + distr_count * 2 :]]
            result = test((WeibullModel, x, start_O), 5)
            for i, res in enumerate(result):
                # res[0] = [list(r) for r in res[0]]
                out.write(f"{ind}\t{i}\t{O}\t{start_O}\t{res}\n")


def Describe(p_source, p_results):
    data = {}
    with open(p_source, "r") as f:
        for l in f:
            splitted = l.split(";")
            ind, distr_count = (int(i) for i in splitted[:2])
            O = []
            for i in range(distr_count):
                O.append(tuple((float(t) for t in splitted[2 + i * 2 : 4 + i * 2])))

            start_O = []
            for i in range(distr_count):
                r = 2 + distr_count * 2 + i * 2
                start_O.append(tuple((float(t) for t in splitted[r : r + 2])))

            x = [float(t) for t in splitted[4 + distr_count * 2 :]]

            data[ind] = types.SimpleNamespace()
            data[ind].ind = i
            data[ind].k = distr_count
            data[ind].O = start_O
            data[ind].x = x
    for p_result in p_results:
        with open(p_result, "r") as f:
            for l in f:
                splitted = l.split("\t")
    return splitted


if __name__ == "__main__":
    run_test(sys.argv[1], sys.argv[2], sys.argv[3])


def Generate(count, p):
    result = ""
    for i in range(count):
        result += f"| .venv/bin/python run_test.py {p} {i} {count} "
    return result[2:]
