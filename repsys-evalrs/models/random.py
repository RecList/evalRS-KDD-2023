import numpy as np
from repsys.helpers import set_seed
from scipy.sparse import csr_matrix

from models.base import BaseModel


class RandomModel(BaseModel):
    def name(self) -> str:
        return "rand"

    def fit(self, training: bool = False) -> None:
        return

    def predict(self, X: csr_matrix, **kwargs):
        X_predict = np.ones(X.shape)
        X_predict[X.nonzero()] = 0

        set_seed(self.config.seed)
        item_ratings = np.random.uniform(size=X.shape)

        X_predict = X_predict * item_ratings

        self._apply_filters(X_predict, **kwargs)

        return X_predict
