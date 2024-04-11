import matplotlib.pyplot as plt
from models.models import Results
import numpy as np


class Tools:
    @property
    def dos(self,results:Results, **kwargs):
        plt.hist(np.array([r.eigenvalues for r in results]).flatten(), **kwargs)
