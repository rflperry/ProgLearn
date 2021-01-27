"""
Main Author: Will LeVine 
Corresponding Email: levinewill@icloud.com
"""
import keras
import numpy as np
from itertools import product
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.random_projection import SparseRandomProjection
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

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
        check_X_y(X, y)
        _, y = np.unique(y, return_inverse=True)

        # more typechecking
        self.network.compile(
            loss=self.loss, optimizer=self.optimizer, **self.compile_kwargs
        )

        self.network.fit(X, keras.utils.to_categorical(y), **self.fit_kwargs)

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
        check_array(X)
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


class ObliqueTreeClassificationTransformer(BaseTransformer):
    """
    A class used to transform data from a category to a specialized representation.

    Parameters
    ----------
    kwargs : dict, default={}
        A dictionary to contain parameters of the tree.

    Attributes
    ----------
    transformer : ObliqueTreeClassifier
        an sklearn compliant oblique decision tree (SPORF)
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
        self.transformer_ = ObliqueTreeClassifier(**self.kwargs).fit(X, y)
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


"""
Authors: Parth Vora and Jay Mandavilli

Oblique Decision Tree (SPORF)
"""
# --------------------------------------------------------------------------
class SplitInfo:
    """
    A class used to store information about a certain split.

    Parameters
    ----------
    feature : int
        The feature which is used for the particular split.
    threshold : float
        The feature value which defines the split, if an example has a value less
        than this threshold for the feature of this split then it will go to the
        left child, otherwise it wil go the right child where these children are
        the children nodes of the node for which this split defines.
    proj_mat : array of shape [n_components, n_features]
        The sparse random projection matrix for this split.
    left_impurity : float
        This is Gini impurity of left side of the split.
    left_idx : array of shape [left_n_samples]
        This is the indices of the nodes that are in the left side of this split.
    left_n_samples : int
        The number of samples in the left side of this split.
    right_impurity : float
        This is Gini impurity of right side of the split.
    right_idx : array of shape [right_n_samples]
        This is the indices of the nodes that are in the right side of this split.
    right_n_samples : int
        The number of samples in the right side of this split.
    no_split : bool
        A boolean specifying if there is a valid split or not. Here an invalid
        split means all of the samples would go to one side.
    improvement : float
        A metric to determine if the split improves the decision tree.
    """

    def __init__(
        self,
        feature,
        threshold,
        proj_mat,
        left_impurity,
        left_idx,
        left_n_samples,
        right_impurity,
        right_idx,
        right_n_samples,
        no_split,
        improvement,
    ):

        self.feature = feature
        self.threshold = threshold
        self.proj_mat = proj_mat
        self.left_impurity = left_impurity
        self.left_idx = left_idx
        self.left_n_samples = left_n_samples
        self.right_impurity = right_impurity
        self.right_idx = right_idx
        self.right_n_samples = right_n_samples
        self.no_split = no_split
        self.improvement = improvement


class ObliqueSplitter:
    """
    A class used to represent an oblique splitter, where splits are done on
    the linear combination of the features.

    Parameters
    ----------
    X : array of shape [n_samples, n_features]
        The input data X is a matrix of the examples and their respective feature
        values for each of the features.
    y : array of shape [n_samples]
        The labels for each of the examples in X.
    proj_dims : int
        The dimensionality of the target projection space.
    density : float
        Ratio of non-zero component in the random projection matrix in the range '(0, 1]'.
    random_state : int
        Controls the pseudo random number generator used to generate the projection matrix.
    workers : int
        The number of cores to parallelize the calculation of Gini impurity.
        Supply -1 to use all cores available to the Process.

    Methods
    -------
    sample_proj_mat(sample_inds)
        This gets the projection matrix and it fits the transform to the samples of interest.
    leaf_label_proba(idx)
        This calculates the label and the probability for that label for a particular leaf
        node.
    score(y_sort, t)
        Finds the Gini impurity for a split.
    _score(self, proj_X, y_sample, i, j)
        Handles array indexing before calculating Gini impurity.
    impurity(idx)
        Finds the impurity for a certain set of samples.
    split(sample_inds)
        Determines the best possible split for the given set of samples.
    """

    def __init__(self, X, y, proj_dims, density, random_state, workers):

        self.X = X
        self.y = y

        self.classes = np.array(np.unique(y), dtype=int)
        self.n_classes = len(self.classes)
        self.indices = np.indices(y.shape)[0]

        self.n_samples = X.shape[0]

        self.proj_dims = proj_dims
        self.density = density
        self.random_state = random_state
        self.workers = workers

    def sample_proj_mat(self, sample_inds):
        """
        Gets the projection matrix and it fits the transform to the samples of interest.

        Parameters
        ----------
        sample_inds : array of shape [n_samples]
            The data we are transforming.

        Returns
        -------
        proj_mat : {ndarray, sparse matrix} of shape (self.proj_dims, n_features)
            The generated sparse random matrix.
        proj_X : {ndarray, sparse matrix} of shape (n_samples, self.proj_dims)
            Projected input data matrix.
        """

        proj_mat = SparseRandomProjection(
            density=self.density,
            n_components=self.proj_dims,
            random_state=self.random_state,
        )

        proj_X = proj_mat.fit_transform(self.X[sample_inds, :])
        return proj_X, proj_mat

    def leaf_label_proba(self, idx):
        """
        Finds the most common label and probability of this label from the samples at
        the leaf node for which this is used on.

        Parameters
        ----------
        idx : array of shape [n_samples]
            The indices of the samples that are at the leaf node for which the label
            and probability need to be found.

        Returns
        -------
        label : int
            The label for any sample that is predicted to be at this node.
        proba : float
            The probability of the predicted sample to have this node's label.
        """

        samples = self.y[idx]
        n = len(samples)
        labels, count = np.unique(samples, return_counts=True)
        most = np.argmax(count)

        label = labels[most]
        proba = count[most] / n

        return label, proba

    # Returns gini impurity for split
    # Expects 0 < t < n
    def score(self, y_sort, t):
        """
        Finds the Gini impurity for the split of interest

        Parameters
        ----------
        y_sort : array of shape [n_samples]
            A sorted array of labels for the examples for which the Gini impurity
            is being calculated.
        t : float
            The threshold determining where to split y_sort.

        Returns
        -------
        gini : float
            The Gini impurity of the split.
        """

        left = y_sort[:t]
        right = y_sort[t:]

        n_left = len(left)
        n_right = len(right)

        left_unique, left_counts = np.unique(left, return_counts=True)
        right_unique, right_counts = np.unique(right, return_counts=True)

        left_counts = left_counts / n_left
        right_counts = right_counts / n_right

        left_gini = 1 - np.sum(np.power(left_counts, 2))
        right_gini = 1 - np.sum(np.power(right_counts, 2))

        gini = (n_left / self.n_samples) * left_gini + (
            n_right / self.n_samples
        ) * right_gini
        return gini

    def _score(self, proj_X, y_sample, i, j):
        """
        Handles array indexing before calculating Gini impurity

        Parameters
        ----------
        proj_X : {ndarray, sparse matrix} of shape (n_samples, self.proj_dims)
            Projected input data matrix.
        y_sample : array of shape [n_samples]
            Labels for sample of data.
        i : float
            The threshold determining where to split y_sort.
        j : float
            The projection dimension to consider.

        Returns
        -------
        gini : float
            The Gini impurity of the split.
        i : float
            The threshold determining where to split y_sort.
        j : float
            The projection dimension to consider.
        """
        # Sort labels by the jth feature
        idx = np.argsort(proj_X[:, j])
        y_sort = y_sample[idx]

        gini = self.score(y_sort, i)

        return gini, i, j

    # Returns impurity for a group of examples
    # expects idx not None
    def impurity(self, idx):
        """
        Finds the actual impurity for a set of samples

        Parameters
        ----------
        idx : array of shape [n_samples]
            The indices of the nodes in the set for which the impurity is being calculated.

        Returns
        -------
        impurity : float
            Actual impurity of split.
        """

        samples = self.y[idx]
        n = len(samples)

        if n == 0:
            return 0

        unique, count = np.unique(samples, return_counts=True)
        count = count / n
        gini = np.sum(np.power(count, 2))

        return 1 - gini

    # Finds the best split
    def split(self, sample_inds):
        """
        Finds the optimal split for a set of samples.

        Parameters
        ----------
        sample_inds : array of shape [n_samples]
            The indices of the nodes in the set for which the best split is found.

        Returns
        -------
        split_info : SplitInfo
            Class holding information about the split.
        """

        # Project the data
        proj_X, proj_mat = self.sample_proj_mat(sample_inds)
        y_sample = self.y[sample_inds]
        n_samples = len(sample_inds)

        # Score matrix
        # No split score is just node impurity
        Q = np.zeros((n_samples, self.proj_dims))
        node_impurity = self.impurity(sample_inds)
        Q[0, :] = node_impurity
        Q[-1, :] = node_impurity

        # Loop through examples and projected features to calculate split scores
        split_iterator = product(range(1, n_samples - 1), range(self.proj_dims))
        scores = Parallel(n_jobs=self.workers)(
            delayed(self._score)(proj_X, y_sample, i, j) for i, j in split_iterator
        )
        for gini, i, j in scores:
            Q[i, j] = gini

        # Identify best split feature, minimum gini impurity
        best_split_ind = np.argmin(Q)
        thresh_i, feature = np.unravel_index(best_split_ind, Q.shape)
        best_gini = Q[thresh_i, feature]

        # Sort samples by the split feature
        feat_vec = proj_X[:, feature]
        idx = np.argsort(feat_vec)

        feat_vec = feat_vec[idx]
        sample_inds = sample_inds[idx]

        # Get the threshold, split samples into left and right
        threshold = feat_vec[thresh_i]
        left_idx = sample_inds[:thresh_i]
        right_idx = sample_inds[thresh_i:]

        left_n_samples = len(left_idx)
        right_n_samples = len(right_idx)

        # See if we have no split
        no_split = left_n_samples == 0 or right_n_samples == 0

        # Evaluate improvement
        improvement = node_impurity - best_gini

        # Evaluate impurities for left and right children
        left_impurity = self.impurity(left_idx)
        right_impurity = self.impurity(right_idx)

        split_info = SplitInfo(
            feature,
            threshold,
            proj_mat,
            left_impurity,
            left_idx,
            left_n_samples,
            right_impurity,
            right_idx,
            right_n_samples,
            no_split,
            improvement,
        )

        return split_info


# --------------------------------------------------------------------------


class Node:
    """
    A class used to represent an oblique node.

    Parameters
    ----------
    None

    Methods
    -------
    None
    """

    def __init__(self):
        self.node_id = None
        self.is_leaf = None
        self.parent = None
        self.left_child = None
        self.right_child = None

        self.feature = None
        self.threshold = None
        self.impurity = None
        self.n_samples = None

        self.proj_mat = None
        self.label = None
        self.proba = None


class StackRecord:
    """
    A class used to keep track of a node's parent and other information about the node and its split.

    Parameters
    ----------
    parent : int
        The index of the parent node.
    depth : int
        The depth at which this node is.
    is_left : bool
        Represents if the node is a left child or not.
    impurity : float
        This is Gini impurity of this node.
    sample_idx : array of shape [n_samples]
        This is the indices of the nodes that are in this node.
    n_samples : int
        The number of samples in this node.

    Methods
    -------
    None
    """

    def __init__(self, parent, depth, is_left, impurity, sample_idx, n_samples):

        self.parent = parent
        self.depth = depth
        self.is_left = is_left
        self.impurity = impurity
        self.sample_idx = sample_idx
        self.n_samples = n_samples


class ObliqueTree:
    """
    A class used to represent a tree with oblique splits.

    Parameters
    ----------
    splitter : class
        The type of splitter for this tree, should be an ObliqueSplitter.
    min_samples_split : int
        Minimum number of samples possible at a node.
    min_samples_leaf : int
        Minimum number of samples possible at a leaf.
    max_depth : int
        Maximum depth allowed for the tree.
    min_impurity_split : float
        Minimum Gini impurity value that must be achieved for a split to occur on the node.
    min_impurity_decrease : float
        Minimum amount Gini impurity value must decrease by for a split to be valid.

    Methods
    -------
    add_node(parent, is_left, impurity, n_samples, is_leaf, feature, threshold, proj_mat, label, proba)
        Adds a node to the existing tree
    build()
        This is what is initially called on to completely build the oblique tree.
    predict(X)
        Finds the final node for each input sample as it passes through the decision tree.
    """

    def __init__(
        self,
        splitter,
        min_samples_split,
        min_samples_leaf,
        max_depth,
        min_impurity_split,
        min_impurity_decrease,
    ):

        # Tree parameters
        # self.n_samples = n_samples
        # self.n_features = n_features
        # self.n_classes = n_classes
        self.depth = 0
        self.node_count = 0
        self.nodes = []

        # Build parameters
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.min_impurity_split = min_impurity_split
        self.min_impurity_decrease = min_impurity_decrease

    def add_node(
        self,
        parent,
        is_left,
        impurity,
        n_samples,
        is_leaf,
        feature,
        threshold,
        proj_mat,
        label,
        proba,
    ):
        """
        Adds a node to the existing oblique tree.

        Parameters
        ----------
        parent : int
            The index of the parent node for the new node being added.
        is_left : bool
            Determines if this new node being added is a left or right child.
        impurity : float
            Impurity of this new node.
        n_samples : int
            Number of samples at this new node.
        is_leaf : bool
            Determines if this new node is a leaf of the tree or an internal node.
        feature : int
            Index of feature on which the split occurs at this node.
        threshold : float
            The threshold feature value for this node determining if a sample will go
            to this node's left of right child. If a sample has a value less than the
            threshold (for the feature of this node) it will go to the left childe,
            otherwise it will go the right child.
        proj_mat : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Projection matrix for this new node.
        label : int
            The label a sample will be given if it is predicted to be at this node.
        proba : float
            The probability a predicted sample has of being the node's label.

        Returns
        -------
        node_id : int
            Index of the new node just added.
        """

        node = Node()
        node.node_id = self.node_count
        node.impurity = impurity
        node.n_samples = n_samples

        # If not the root node, set parents
        if self.node_count > 0:
            node.parent = parent
            if is_left:
                self.nodes[parent].left_child = node.node_id
            else:
                self.nodes[parent].right_child = node.node_id

        # Set node parameters
        if is_leaf:
            node.is_leaf = True
            node.label = label
            node.proba = proba
        else:
            node.is_leaf = False
            node.feature = feature
            node.threshold = threshold
            node.proj_mat = proj_mat

        self.node_count += 1
        self.nodes.append(node)

        return node.node_id

    def build(self):
        """
        Builds the oblique tree.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        # Initialize, add root node
        stack = []
        root = StackRecord(
            0,
            1,
            False,
            self.splitter.impurity(self.splitter.indices),
            self.splitter.indices,
            self.splitter.n_samples,
        )
        stack.append(root)

        # Build tree
        while len(stack) > 0:

            # Pop a record off the stack
            cur = stack.pop()

            # Evaluate if it is a leaf
            is_leaf = (
                cur.depth >= self.max_depth
                or cur.n_samples < self.min_samples_split
                or cur.n_samples < 2 * self.min_samples_leaf
                or cur.impurity <= self.min_impurity_split
            )

            # Split if not
            if not is_leaf:
                split = self.splitter.split(cur.sample_idx)

                is_leaf = (
                    is_leaf
                    or split.no_split
                    or split.improvement <= self.min_impurity_decrease
                )

            # Add the node to the tree
            if is_leaf:

                label, proba = self.splitter.leaf_label_proba(cur.sample_idx)

                node_id = self.add_node(
                    cur.parent,
                    cur.is_left,
                    cur.impurity,
                    cur.n_samples,
                    is_leaf,
                    None,
                    None,
                    None,
                    label,
                    proba,
                )

            else:
                node_id = self.add_node(
                    cur.parent,
                    cur.is_left,
                    cur.impurity,
                    cur.n_samples,
                    is_leaf,
                    split.feature,
                    split.threshold,
                    split.proj_mat,
                    None,
                    None,
                )

            # Push the right and left children to the stack if applicable
            if not is_leaf:

                right_child = StackRecord(
                    node_id,
                    cur.depth + 1,
                    False,
                    split.right_impurity,
                    split.right_idx,
                    split.right_n_samples,
                )
                stack.append(right_child)

                left_child = StackRecord(
                    node_id,
                    cur.depth + 1,
                    True,
                    split.left_impurity,
                    split.left_idx,
                    split.left_n_samples,
                )
                stack.append(left_child)

            if cur.depth > self.depth:
                self.depth = cur.depth

    def predict(self, X):
        """
        Predicts final nodes of samples given.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input array for which predictions are made.

        Returns
        -------
        predictions : array of shape [n_samples]
            Array of the final node index for each input prediction sample.
        """

        predictions = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            cur = self.nodes[0]
            while cur is not None and not cur.is_leaf:
                proj_X = cur.proj_mat.transform(X)
                if proj_X[i, cur.feature] < cur.threshold:
                    id = cur.left_child
                    cur = self.nodes[id]
                else:
                    id = cur.right_child
                    cur = self.nodes[id]

            predictions[i] = cur.node_id

        return predictions


# --------------------------------------------------------------------------

""" Class for Oblique Tree """


class ObliqueTreeClassifier(BaseEstimator):
    """
    A class used to represent a classifier that uses an oblique decision tree.

    Parameters
    ----------
    max_depth : int
        Maximum depth allowed for oblique tree.
    min_samples_split : int
        Minimum number of samples possible at a node.
    min_samples_leaf : int
        Minimum number of samples possible at a leaf.
    random_state : int
        Maximum depth allowed for the tree.
    min_impurity_decrease : float
        Minimum amount Gini impurity value must decrease by for a split to be valid.
    min_impurity_split : float
        Minimum Gini impurity value that must be achieved for a split to occur on the node.
    feature_combinations : float
        The feature combinations to use for the oblique split.
    density : float
        Density estimate.
    workers : int, optional (default: -1)
        The number of cores to parallelize the calculation of Gini impurity.
        Supply -1 to use all cores available to the Process.

    Methods
    -------
    fit(X,y)
        Fits the oblique tree to the training samples.
    apply(X)
        Calls on the predict function from the oblique tree for the test samples.
    predict(X)
        Gets the prediction labels for the test samples.
    predict_proba(X)
        Gets the probability of the prediction labels for the test samples.
    predict_log_proba(X)
        Gets the log of the probability of the prediction labels for the test samples.
    """

    def __init__(
        self,
        *,
        # criterion="gini",
        # splitter=None,
        max_depth=np.inf,
        min_samples_split=2,
        min_samples_leaf=1,
        # min_weight_fraction_leaf=0,
        # max_features="auto",
        # max_leaf_nodes=None,
        random_state=None,
        min_impurity_decrease=0,
        min_impurity_split=0,
        # class_weight=None,
        # ccp_alpha=0.0,
        # New args
        feature_combinations=1.5,
        density=0.5,
        workers=-1,
    ):

        # self.criterion=criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        # self.min_weight_fraction_leaf=min_weight_fraction_leaf
        # self.max_features=max_features
        # self.max_leaf_nodes=max_leaf_nodes
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        # self.class_weight=class_weight
        # self.ccp_alpha=ccp_alpha

        self.feature_combinations = feature_combinations
        self.density = density
        self.workers = workers

    def fit(self, X, y):
        """
        Predicts final nodes of samples given.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The training samples.
        y : array of shape [n_samples]
            Labels for the training samples.

        Returns
        -------
        ObliqueTreeClassifier
            The fit classifier.
        """

        self.proj_dims = int(np.ceil(X.shape[1]) / self.feature_combinations)
        splitter = ObliqueSplitter(
            X, y, self.proj_dims, self.density, self.random_state, self.workers
        )

        self.tree = ObliqueTree(
            splitter,
            self.min_samples_split,
            self.min_samples_leaf,
            self.max_depth,
            self.min_impurity_split,
            self.min_impurity_decrease,
        )
        self.tree.build()
        return self

    def apply(self, X):
        """
        Gets predictions form the oblique tree for the test samples.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The testing samples.

        Returns
        -------
        pred_nodes : array of shape[n_samples]
            The indices for each test sample's final node in the oblique tree.
        """

        pred_nodes = self.tree.predict(X).astype(int)
        return pred_nodes

    def predict(self, X):
        """
        Determines final label predictions for each sample in the test data.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The testing samples.

        Returns
        -------
        preds : array of shape[n_samples]
            The predictions (labels) for each testing sample.
        """

        preds = np.zeros(X.shape[0])
        pred_nodes = self.apply(X)
        for k in range(len(pred_nodes)):
            id = pred_nodes[k]
            preds[k] = self.tree.nodes[id].label

        return preds

    def predict_proba(self, X):
        """
        Determines probabilities of the final label predictions for each sample in the test data.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The testing samples.

        Returns
        -------
        preds : array of shape[n_samples]
            The probabilities of the predictions (labels) for each testing sample.
        """

        preds = np.zeros(X.shape[0])
        pred_nodes = self.apply(X)
        for k in range(len(preds)):
            id = pred_nodes[k]
            preds[k] = self.tree.nodes[id].proba

        return preds

    def predict_log_proba(self, X):
        """
        Determines log of the probabilities of the final label predictions for each sample in the test data.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The testing samples.

        Returns
        -------
        preds : array of shape[n_samples]
            The log of the probabilities of the predictions (labels) for each testing sample.
        """

        proba = self.predict_proba(X)
        for k in range(len(proba)):
            proba[k] = np.log(proba[k])

        return proba
