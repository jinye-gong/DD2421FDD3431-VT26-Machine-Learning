import random
import numpy as np
import matplotlib.pyplot as plt
import monkdata as m
import dtree as d

def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

def prune(tree, valset):
    best_tree = tree
    best_acc = d.check(best_tree, valset)

    while True:
        candidates = d.allPruned(best_tree)
        cand_tree = max(candidates, key=lambda t: d.check(t, valset))
        cand_acc = d.check(cand_tree, valset)

        if cand_acc >= best_acc:
            best_tree, best_acc = cand_tree, cand_acc
        else:
            break

    return best_tree

def evaluate_pruning(trainset, testset, fractions, runs=100, seed=0):
    random.seed(seed)
    results_mean = []
    results_std = []

    for frac in fractions:
        errors = []
        for _ in range(runs):
            tr, val = partition(trainset, frac)
            t = d.buildTree(tr, m.attributes)     # full tree on training part
            tp = prune(t, val)                    # prune using validation part
            acc = d.check(tp, testset)
            err = 1 - acc
            errors.append(err)

        errors = np.array(errors)
        results_mean.append(errors.mean())
        results_std.append(errors.std(ddof=1))    # sample std

    return np.array(results_mean), np.array(results_std)

if __name__ == "__main__":
    fractions = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    runs = 100

    m1_mean, m1_std = evaluate_pruning(m.monk1, m.monk1test, fractions, runs=runs, seed=1)
    m3_mean, m3_std = evaluate_pruning(m.monk3, m.monk3test, fractions, runs=runs, seed=2)

    # 画图（带误差条 + 数据点 + 图例 + 轴标签）
    plt.figure()
    plt.errorbar(fractions, m1_mean, yerr=m1_std, marker='o', capsize=4, label='MONK-1')
    plt.errorbar(fractions, m3_mean, yerr=m3_std, marker='o', capsize=4, label='MONK-3')
    plt.xlabel('fraction (training split)')
    plt.ylabel('test classification error')
    plt.title('Reduced Error Pruning: test error vs fraction')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.show()

    # 同时把数值打印出来，方便你写报告表格
    print("fraction | monk1 mean±std | monk3 mean±std")
    for f, a, b, c, dstd in zip(fractions, m1_mean, m1_std, m3_mean, m3_std):
        print(f"{f:.1f} | {a:.6f} ± {b:.6f} | {c:.6f} ± {dstd:.6f}")

