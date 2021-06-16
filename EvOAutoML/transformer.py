from skmultiflow.transform import MissingValuesCleaner, WindowedMinmaxScaler, WindowedStandardScaler
from skmultiflow.utils import get_dimensions, FastBuffer
import numpy as np


class ExtendedMissingValuesCleaner(MissingValuesCleaner):

    def __init__(self, missing_value=np.nan, strategy='zero', window_size=200, new_value=1):
        super().__init__()
        if isinstance(missing_value, list):
            self.missing_value = missing_value
        else:
            self.missing_value = [missing_value]
        self.strategy = strategy
        self.window_size = window_size
        self.window = FastBuffer(max_size=self.window_size)
        self.new_value = new_value

        self.__configure()

    def __configure(self):
        if self.strategy in ['mean', 'median', 'mode']:
            self.window = FastBuffer(max_size=self.window_size)

    def fit(self,X,y):
        return self

    def transform(self, X):
        """ transform

        Does the transformation process in the samples in X.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            The sample or set of samples that should be transformed.

        """
        r, c = get_dimensions(X)
        for i in range(r):
            if self.strategy in ['mean', 'median', 'mode']:
                self.window.add_element([X[i][:]])
            for j in range(c):
                if X[i][j] in self.missing_value or np.isnan(X[i][j]):
                    X[i][j] = self._get_substitute(j)

        return X

class ExtendedWindowedMinmaxScaler(WindowedMinmaxScaler):
    def fit(self, X,y):
        return self

class ExtendedWindowedStandardScaler(WindowedStandardScaler):
    def fit(self, X, y):
        return self
