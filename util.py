import numpy as np


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def merge(a, s=2):
    return np.array([np.sum(a[s*i:s*(i+1)]) / s for i in range(a.size // s)])
