import numpy as np


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def merge(a, s=1):
    return np.array([np.sum(a[s*i:s*(i+1)]) / s for i in range(a.size // s)])

    
def get_index_for_value(array: np.array, val: float) -> int:
    return np.argmin(abs(array- val))