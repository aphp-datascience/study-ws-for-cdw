from typing import List
from metal.label_model import LabelModel
from wedsak.registry import registry
from wedsak.label_model.base import LabelModelProtocol


@registry.label_model.register("MetalLabelModel")
class MetalLabelModelWrapper(LabelModel, LabelModelProtocol):
    def __init__(
        self,
        k=2,
        seed=123,
        n_epochs: int = 10000,
        mu_epochs: int = 10000,
        log_train_every: int = 1000,
        class_balance_train: List[float] = [0.97, 0.03],
        lr: float = 0.0001,
        optimizer: str = "sgd",
        batch_size: int = 10000,
        deps: list = [],
        **kwargs,
    ):
        """
        n_epochs : int, default=10000
            Number of epochs for training the label model.
        mu_epochs : int, default=10000
            Number of epochs for the mu step in label model training.
        log_train_every : int, default=1000
            Frequency of logging during training.
        deps : list, default=[]
            Dependencies for the label model training.
        class_balance_train : list, default=[0.97, 0.03]
            Class balance for the training set.
        lr : float, default=0.0001
            Learning rate for the label model training.
        k : int, default=2
            Number of classes for the label model.
        seed : int, default=123
            Random seed for reproducibility.
        optimizer : str, default="sgd"
            Optimizer to use for training the label model.
        batch_size : int, default=10000
            Batch size for training the label model.
        """
        super().__init__(k, seed=seed, **kwargs)
        self.name = "metal"
        self.n_epochs = n_epochs
        self.mu_epochs = mu_epochs
        self.log_train_every = log_train_every
        self.class_balance_train = class_balance_train
        self.lr = lr
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.deps = deps

    def fit(
        self,
        L_train,
        **kwargs,
    ):
        kwargs_train = dict(optimizer=self.optimizer, batch_size=self.batch_size)
        kwargs_train.update(kwargs)

        self.train_model(
            L_train,
            n_epochs=self.n_epochs,
            log_train_every=self.log_train_every,
            mu_epochs=self.mu_epochs,
            deps=self.deps,
            lr=self.lr,
            class_balance=self.class_balance_train,
            **kwargs_train,
        )

    def preprocess(
        self,
    ):
        pass

    def postprocess(
        self,
    ):
        pass

    def _predict_probs(self, L):
        probs = self.predict_proba(L)
        return probs

    def _predict(self, L, break_ties="random", return_probs=False, **kwargs):
        return self.predict(L, break_ties, return_probs, **kwargs)

    def predict_probs(self, L_train=None, L_dev=None, **kwargs):
        self.preprocess()
        if L_train is not None:
            probs_train = self._predict_probs(L_train)
        else:
            probs_train = None
        if L_dev is not None:
            probs_dev = self._predict_probs(L_dev)
        else:
            probs_dev = None
        self.postprocess()

        return probs_train, probs_dev

    def predict_label(
        self,
        L_train=None,
        L_dev=None,
        break_ties="random",
        return_probs=False,
        **kwargs,
    ):
        self.preprocess()
        if L_train is not None:
            preds_train = self._predict(L_train, break_ties, return_probs, **kwargs)
        else:
            preds_train = None
        if L_dev is not None:
            preds_dev = self._predict(L_dev, break_ties, return_probs, **kwargs)
        else:
            preds_dev = None
        self.postprocess()

        return preds_train, preds_dev
