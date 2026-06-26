from hyperlm import HyperLabelModel
import scipy
import numpy as np
from pathlib import Path
from wedsak.registry import registry
from wedsak.label_model.base import LabelModelProtocol


@registry.label_model.register("HyperLabelModel")
class HyperLabelModelWrapper(HyperLabelModel, LabelModelProtocol):
    def __init__(
        self,
        checkpoint_path: str = "~/hyper_label_model/HLM_state_dict",
        k: int = 2,
        device: str = "cpu",
        **kwargs,
    ):
        """
        Wrapper for HyperLabelModel from hyperlm package.
        Note in X, -1 represents abstention, 0 and 1 represent classes.
        Each row of X includes the weak labels for a data point,
        and each column of X includes the weak labels from a labeling function (LF).
        """
        super().__init__(
            checkpoint_path=Path(checkpoint_path).expanduser(), device=device, **kwargs
        )
        self.name = "hyper_label"
        self.cardinality = k

    def fit(self, **kwargs):
        pass

    def preprocess(self, L_train=None, L_dev=None):
        if L_dev is not None:
            L_dev_dense = self._get_dense_matrix(L_dev)
        else:
            L_dev_dense = np.empty(
                shape=(0, self.cardinality),
                dtype=np.int32,
            )

        dev_length = L_dev_dense.shape[0]

        if L_train is not None:
            L_train_dense = self._get_dense_matrix(L_train)
        else:
            L_train_dense = np.empty(
                shape=(0, self.cardinality),
                dtype=np.int32,
            )

        # Concatenate train and dev for inference
        L_all = np.concatenate([L_dev_dense, L_train_dense], axis=0)

        return L_all, dev_length

    def postprocess(self, pred_all, dev_length):
        pred_dev = pred_all[:dev_length]
        pred_train = pred_all[dev_length:]
        return pred_train, pred_dev

    @staticmethod
    def _get_dense_matrix(L):
        if scipy.sparse.issparse(L):
            L_dense = L.todense()
        else:
            L_dense = L
        return L_dense

    def predict_label(self, L_train=None, L_dev=None, **kwargs):
        L_all, dev_length = self.preprocess(L_train, L_dev)

        pred_all = self.infer(
            L_all,
            return_probs=False,
        )

        pred_train, pred_dev = self.postprocess(pred_all, dev_length)
        return pred_train, pred_dev

    def predict_probs(self, L_train=None, L_dev=None, **kwargs):
        L_all, dev_length = self.preprocess(L_train, L_dev)

        probs_all = self.infer(
            L_all,
            return_probs=True,
        )
        probs_train, probs_dev = self.postprocess(probs_all, dev_length)
        return probs_train, probs_dev
