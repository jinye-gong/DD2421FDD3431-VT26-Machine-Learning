import monkdata as m
import dtree as d

print("Entropy of training sets:")
print(f"MONK-1: {d.entropy(m.monk1):.6f}")
print(f"MONK-2: {d.entropy(m.monk2):.6f}")
print(f"MONK-3: {d.entropy(m.monk3):.6f}")

