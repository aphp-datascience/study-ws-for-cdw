from typing import Protocol, Optional, Tuple, runtime_checkable
import numpy as np


@runtime_checkable
class LabelModelProtocol(Protocol):
    def fit(self, L_train: np.ndarray, **kwargs) -> None:
        pass

    def predict_label(
        self, L_train: np.ndarray, L_dev: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def predict_probs(
        self, L_train: np.ndarray, L_dev: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass
