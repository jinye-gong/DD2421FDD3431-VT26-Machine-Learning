import monkdata as m
import dtree as d

def gains(dataset, name):
    gs = [d.averageGain(dataset, a) for a in m.attributes]
    print(name)
    for i, g in enumerate(gs, start=1):
        print(f"  a{i}: {g:.12f}")
    best_i = max(range(len(gs)), key=lambda i: gs[i])
    print(f"  -> best attribute at root: a{best_i+1}\n")

gains(m.monk1, "MONK-1")
gains(m.monk2, "MONK-2")
gains(m.monk3, "MONK-3")

