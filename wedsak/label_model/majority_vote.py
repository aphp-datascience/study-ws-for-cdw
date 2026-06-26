import numpy as np
from wedsak.registry import registry
from wedsak.label_model.base import LabelModelProtocol


@registry.label_model.register("MajorityVoteLabelModel")
class MajorityVoteLabelModel(LabelModelProtocol):
    """Majority vote label model."""

    def __init__(self, k: int = 2, abstention_value: int = -1):
        self.cardinality = k
        self.abstention_value = abstention_value
        self.name = "majority_vote"

    def fit(self, **kwargs):
        pass

    def _predict_probs(self, L: np.ndarray, **kwargs) -> np.ndarray:
        """Predict probabilities using majority vote. (extracted from Snorkel)

        Assign vote by calculating majority vote across all labeling functions.
        In case of ties, non-integer probabilities are possible.

        Parameters
        ----------
        L
            An [n, m] matrix of labels
        abstetion_value: int, {-1 or 0}

        Returns
        -------
        np.ndarray
            A [n, k] array of probabilistic labels

        Example
        -------
        >>> L = np.array([[0, 0, -1], [-1, 0, 1], [1, -1, 0]])
        >>> maj_voter = MajorityLabelVoter()
        >>> maj_voter.predict_proba(L)
        array([[1. , 0. ],
               [0.5, 0.5],
               [0.5, 0.5]])
        """
        n, m = L.shape
        Y_p = np.zeros((n, self.cardinality))
        for i in range(n):
            counts = np.zeros(self.cardinality)
            for j in range(m):
                if L[i, j] != self.abstention_value:
                    if self.abstention_value == -1:
                        counts[L[i, j]] += 1
                    elif self.abstention_value == 0:
                        counts[L[i, j] - 1] += 1
                    else:
                        raise ValueError("abstention_value must be -1 or 0")
            Y_p[i, :] = np.where(counts == max(counts), 1, 0)
        Y_p /= Y_p.sum(axis=1).reshape(-1, 1)
        return Y_p

    def predict_probs(self, L_train=None, L_dev=None, **kwargs):
        probs_train = (
            self._predict_probs(L_train, **kwargs) if L_train is not None else None
        )
        probs_dev = self._predict_probs(L_dev, **kwargs) if L_dev is not None else None
        return probs_train, probs_dev

    def break_ties(
        self,
    ):
        return np.random.dirichlet(alpha=[1] * self.cardinality, size=1)

    def chosen_class(self, probs):
        return np.where(probs == 1 / self.cardinality, self.break_ties(), probs).argmax(
            axis=1
        )

    def predict_label(self, L_train=None, L_dev=None, **kwargs):
        preds_train = None
        preds_dev = None

        probs_train = (
            self._predict_probs(L_train, **kwargs) if L_train is not None else None
        )
        probs_dev = self._predict_probs(L_dev, **kwargs) if L_dev is not None else None

        if probs_train is not None:
            preds_train = self.chosen_class(probs_train)
        if probs_dev is not None:
            preds_dev = self.chosen_class(probs_dev)

        return preds_train, preds_dev
