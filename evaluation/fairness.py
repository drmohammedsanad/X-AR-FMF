import numpy as np

def exposure_disparity(recs, item_groups):
    counts = {0:0, 1:0}
    for i in recs:
        counts[item_groups.get(i,0)] += 1
    return abs(counts[0] - counts[1])
