import monkdata as m
import dtree as d

datasets = [
    ("MONK-1", m.monk1, m.monk1test),
    ("MONK-2", m.monk2, m.monk2test),
    ("MONK-3", m.monk3, m.monk3test),
]

print("| Dataset | Train acc | Train err | Test acc | Test err |")
print("|---|---:|---:|---:|---:|")

for name, train, test in datasets:
    t = d.buildTree(train, m.attributes)     # full tree
    train_acc = d.check(t, train)
    test_acc  = d.check(t, test)
    print(f"| {name} | {train_acc:.6f} | {1-train_acc:.6f} | {test_acc:.6f} | {1-test_acc:.6f} |")

