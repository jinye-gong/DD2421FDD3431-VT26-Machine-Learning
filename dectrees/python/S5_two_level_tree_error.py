import monkdata as m
import dtree as d

# MONK 各属性的取值范围（用于枚举分支）
VALUES = {
    1: [1, 2, 3],
    2: [1, 2, 3],
    3: [1, 2],
    4: [1, 2, 3],
    5: [1, 2, 3, 4],
    6: [1, 2],
}

def a_idx(attr):
    """attr -> 1..6"""
    return m.attributes.index(attr) + 1

def a_name(attr):
    return f"a{a_idx(attr)}"

def is_pure(dataset, eps=1e-12):
    """用熵接近 0 判断是否纯"""
    return d.entropy(dataset) < eps

def best_attr(dataset, excluded_attr):
    """在 dataset 上找除 excluded_attr 外 IG 最大的属性"""
    candidates = [a for a in m.attributes if a != excluded_attr]
    gains = [(a, d.averageGain(dataset, a)) for a in candidates]
    best = max(gains, key=lambda x: x[1])
    return best[0], best[1], gains

root = m.attributes[4]  # a5

print("=== MONK-1: next-level gains after splitting on a5 ===")
print(f"Root attribute: {a_name(root)}\n")

# 1) 按 a5 分裂
subsets = {}
for v in VALUES[a_idx(root)]:
    Sv = d.select(m.monk1, root, v)
    subsets[v] = Sv
    print(f"Subset {a_name(root)}={v}: size={len(Sv)}, entropy={d.entropy(Sv):.6f}")

    if len(Sv) == 0:
        print("  (empty subset)\n")
        continue

    if is_pure(Sv):
        print(f"  Pure -> leaf class (mostCommon) = {d.mostCommon(Sv)}\n")
        continue

    # 2) 计算下一层的信息增益并选最大
    bestA, bestG, gains = best_attr(Sv, root)
    for a, g in gains:
        print(f"  gain({a_name(a)}) = {g:.12f}")
    print(f"  -> best next-level attribute: {a_name(bestA)} (gain={bestG:.12f})\n")

print("\n=== Two-level tree (manual, depth=2; leaves by mostCommon) ===")
print(f"Root: {a_name(root)}")

# 3) 输出两层树，并把第二次分裂后的叶子用 mostCommon 标记
for v in VALUES[a_idx(root)]:
    Sv = subsets[v]
    if len(Sv) == 0:
        continue

    if is_pure(Sv):
        print(f"  {a_name(root)}={v} -> Leaf: {d.mostCommon(Sv)}")
        continue

    bestA, bestG, _ = best_attr(Sv, root)
    print(f"  {a_name(root)}={v} -> Test {a_name(bestA)}")

    for vv in VALUES[a_idx(bestA)]:
        Svv = d.select(Sv, bestA, vv)
        if len(Svv) == 0:
            continue
        print(f"     {a_name(bestA)}={vv} -> Leaf: {d.mostCommon(Svv)}")

# 4) 用 buildTree 的 depth=2 输出对比
print("\n=== ID3 buildTree(monk1, attributes, depth=2) ===")
t2 = d.buildTree(m.monk1, m.attributes, 2)
print(t2)



try:
    import drawtree_qt5 as dt
    print("\n正在启动图形界面绘制决策树...")
    dt.drawTree(t2)
except ImportError:
    print("错误: 无法导入 PyQt5 或 drawtree_qt5。请确保已安装 PyQt5 (pip install PyQt5)。")
except Exception as e:
    print(f"画图时发生错误: {e}")







