"""
Main Author: Will LeVine 
Corresponding Email: levinewill@icloud.com
"""
import numpy as np

from sklearn.tree import DecisionTreeClassifier

from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
)

import keras as keras
import tensorflow as tf

from .base import BaseTransformer


class NeuralClassificationTransformer(BaseTransformer):
    """
    A class used to transform data from a category to a specialized representation.

    Parameters
    ----------
    network : object
        A neural network used in the classification transformer.

    euclidean_layer_idx : int
        An integer to represent the final layer of the transformer.

    optimizer : str or keras.optimizers instance
        An optimizer used when compiling the neural network.

    loss : str, default="categorical_crossentropy"
        A loss function used when compiling the neural network.

    pretrained : bool, default=False
        A boolean used to identify if the network is pretrained.

    compile_kwargs : dict, default={"metrics": ["acc"]}
        A dictionary containing metrics for judging network performance.

    fit_kwargs : dict, default={
                "epochs": 100,
                "callbacks": [keras.callbacks.EarlyStopping(patience=5, monitor="val_acc")],
                "verbose": False,
                "validation_split": 0.33,
            },
        A dictionary to hold epochs, callbacks, verbose, and validation split for the network.

    Attributes
    ----------
    encoder_ : object
        A Keras model with inputs and outputs based on the network attribute.
        Output layers are determined by the euclidean_layer_idx parameter.
    """

    def __init__(
        self,
        network,
        euclidean_layer_idx,
        optimizer,
        loss="categorical_crossentropy",
        pretrained=False,
        compile_kwargs={"metrics": ["acc"]},
        fit_kwargs={
            "epochs": 100,
            "callbacks": [keras.callbacks.EarlyStopping(patience=5, monitor="val_acc")],
            "verbose": False,
            "validation_split": 0.33,
        },
    ):
        self.network = keras.models.clone_model(network)
        self.encoder_ = keras.models.Model(
            inputs=self.network.inputs,
            outputs=self.network.layers[euclidean_layer_idx].output,
        )
        self.pretrained = pretrained
        self.optimizer = optimizer
        self.loss = loss
        self.compile_kwargs = compile_kwargs
        self.fit_kwargs = fit_kwargs

    def fit(self, X, y):
        """
        Fits the transformer to data X with labels y.

        Parameters
        ----------
        X : ndarray
            Input data matrix.
        y : ndarray
            Output (i.e. response data matrix).

        Returns
        -------
        self : NeuralClassificationTransformer
            The object itself.
        """
        #check_X_y(X, y)
        _, y = np.unique(y, return_inverse=True)
        y = keras.utils.to_categorical(y)

        # more typechecking
        self.network.compile(
            loss=self.loss, optimizer=self.optimizer, **self.compile_kwargs
        )

        fit_kwargs = self.fit_kwargs.copy()
        validation_split = fit_kwargs.pop('validation_split')
        batch_size = fit_kwargs.pop('batch_size')

        train_gen = tf.data.Dataset.from_generator(
            lambda: minibatch(X, y, batch_size, validation_split),
            output_signature=(
                tf.TensorSpec(shape=(batch_size, *X.shape[1:])),
                tf.TensorSpec(shape=(batch_size, y.shape[1])),
            ))

        valid_gen = tf.data.Dataset.from_generator(
            lambda: minibatch(X, y, batch_size, validation_split, validation=True),
            output_signature=(
                tf.TensorSpec(shape=(batch_size, *X.shape[1:])),
                tf.TensorSpec(shape=(batch_size, y.shape[1])),
            ))

        self.network.fit(train_gen, validation_data=valid_gen, **fit_kwargs)

        return self

    def transform(self, X):
        """
        Performs inference using the transformer.

        Parameters
        ----------
        X : ndarray
            Input data matrix.

        Returns
        -------
        X_transformed : ndarray
            The transformed input.

        Raises
        ------
        NotFittedError
            When the model is not fitted.
        """
        check_is_fitted(self)
        #check_array(X)
        return self.encoder_.predict(X)


class TreeClassificationTransformer(BaseTransformer):
    """
    A class used to transform data from a category to a specialized representation.

    Parameters
    ----------
    kwargs : dict, default={}
        A dictionary to contain parameters of the tree.

    Attributes
    ----------
    transformer : sklearn.tree.DecisionTreeClassifier
        an internal sklearn DecisionTreeClassifier
    """

    def __init__(self, kwargs={}):
        self.kwargs = kwargs

    def fit(self, X, y):
        """
        Fits the transformer to data X with labels y.

        Parameters
        ----------
        X : ndarray
            Input data matrix.
        y : ndarray
            Output (i.e. response data matrix).

        Returns
        -------
        self : TreeClassificationTransformer
            The object itself.
        """
        X, y = check_X_y(X, y)
        self.transformer_ = DecisionTreeClassifier(**self.kwargs).fit(X, y)
        return self

    def transform(self, X):
        """
        Performs inference using the transformer.

        Parameters
        ----------
        X : ndarray
            Input data matrix.

        Returns
        -------
        X_transformed : ndarray
            The transformed input.

        Raises
        ------
        NotFittedError
            When the model is not fitted.
        """
        check_is_fitted(self)
        X = check_array(X)
        return self.transformer_.apply(X)

def minibatch(X_train, y_train, batch_size, validation_split, validation=False):
    n_tot = X_train.shape[0]
    n_samp = int(n_tot * (1 - validation_split))
    if validation:
        n_samp = n_tot - n_samp
    for _ in range(int(n_samp / batch_size)):
        minibatch_indices = None
        while minibatch_indices is None or min(np.unique(y_train[minibatch_indices], axis=0, return_counts=True)[1]) == 1:   
            minibatch_indices = np.random.choice(n_samp, size=batch_size)
            if validation:
                minibatch_indices += (n_tot - n_samp)

        yield X_train[minibatch_indices], y_train[minibatch_indices]
