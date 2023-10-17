import numpy as np
import torch
def argmax_histogram(data, bins):
    counts, bin_edges = np.histogram(data, bins)
    max_bin_index = np.argmax(counts)
    max_bin = 0.5 * (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1])
    return max_bin


tensor = torch.rand(10, 224, 224)


def max_hist(heatmaps):
    bins = heatmaps.size()[0]
    his_res = torch.zeros(224, 224)    
    for y in range(heatmaps.shape[1]):
        for z in  range(heatmaps.shape[2]):
            his_res[y,z] = argmax_histogram(heatmaps[:,y,z], bins)
    
    return his_res

print(max_hist(tensor).size())