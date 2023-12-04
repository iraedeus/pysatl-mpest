# takes txt file
# test_count k min_k max_k min_l max_l min_k_start max_k_start min_l_start max_l_start sample_count_for_each

import numpy
import random
from scipy.stats import weibull_min
import sys


def generate_distr(min_k, max_k, min_l, max_l):
    k = random.uniform(min_k, max_k)


def generate(t):
    ind = 0
    with open(t, "r") as f:
        with open(t + "test.txt", "w") as o:
            for l in f:
                splitted = l.split()
                test_count, distr_count = (int(i) for i in splitted[:2])
                (
                    min_k,
                    max_k,
                    min_l,
                    max_l,
                    min_k_start,
                    max_k_start,
                    min_l_start,
                    max_l_start,
                ) = (float(i) for i in splitted[2:10])
                counts = [int(i) for i in splitted[10:]]
                for _ in range(test_count):
                    output = f"{ind};{distr_count};"
                    x = []
                    for i in range(distr_count):
                        k = random.uniform(min_k, max_k)
                        l = random.uniform(min_l, max_l)
                        output += f"{k};{l};"
                        x += list(weibull_min.rvs(k, loc=0, scale=l, size=counts[i]))
                    for i in range(distr_count):
                        k = random.uniform(min_k_start, max_k_start)
                        l = random.uniform(min_l_start, max_l_start)
                        output += f"{k};{l};"
                    random.shuffle(x)
                    for sample in x:
                        output += f"{sample};"
                    o.write(output[:-1])
                    o.write("\n")
                    ind += 1


if __name__ == "__main__":
    generate(sys.argv[1])
