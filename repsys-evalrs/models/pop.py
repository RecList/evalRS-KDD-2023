import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler

from models.base import BaseModel


class TopPopular(BaseModel):
    def __init__(self):
        self.item_ratings = None
        self.scaler = MinMaxScaler()

    def name(self) -> str:
        return "pop"

    def fit(self, training: bool = False) -> None:
        X = self.dataset.get_train_data()

        item_popularity = np.asarray(X.sum(axis=0)).reshape(-1, 1)
        item_ratings = self.scaler.fit_transform(item_popularity)

        self.item_ratings = item_ratings.reshape(1, -1)

    def predict(self, X: csr_matrix, **kwargs):
        X_predict = np.ones(X.shape)
        X_predict[X.nonzero()] = 0

        X_predict = X_predict * self.item_ratings

        self._apply_filters(X_predict, **kwargs)

        return X_predict
