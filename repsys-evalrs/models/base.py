import os
from abc import ABC

import numpy as np
from repsys import Model
from repsys.ui import Select


class BaseModel(Model, ABC):
    def _checkpoint_path(self, extension: str = "npy"):
        if extension.startswith('.'):
            extension = extension[1:]
        return os.path.join("./.repsys_checkpoints", f"{self.name()}.{extension}")

    def _create_checkpoints_dir(self):
        dir_path = os.path.dirname(self._checkpoint_path())
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

    def _mask_items(self, X_predict, item_indices):
        mask = np.ones(self.dataset.items.shape[0], dtype=np.bool)
        mask[item_indices] = 0
        X_predict[:, mask] = 0

    def _filter_items(self, X_predict, col, value):
        if value:
            indices = self.dataset.filter_items_by_tags(col, [value])
            self._mask_items(X_predict, indices)

    def _apply_filters(self, X_predict, **kwargs):
        self._filter_items(X_predict, "emotion_tags", kwargs.get("emotion_tags"))
        self._filter_items(X_predict, "social_tags", kwargs.get("social_tags"))

    def web_params(self):
        return {
            "emotion_tags": Select(options=self.dataset.tags.get("emotion_tags")),
            "social_tags": Select(options=self.dataset.tags.get("social_tags")),
        }
