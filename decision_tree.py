# default tree packages
import numpy as np
import pandas as pd
# quality of life packages
from itertools import pairwise
from typing import Tuple, Optional
# displays a nice graph
from graphviz import Digraph
from colorsys import rgb_to_hsv, hsv_to_rgb

from node import Node
from collections import Counter


class DecisionTree:

    @staticmethod
    def _entropy(data, tolerance: float = 1e-8) -> float:
        """
        Calculate the entropy of an array of data.

        Parameters:
            data: Array of some data.
            tolerance (float): Tolerance for entropy threshold.

        Returns:
            float: Entropy value.
        """

        samples = len(data)
        if samples == 0:
            return 0.0
        frequencies = np.array(list(Counter(data).values()))
        total = np.dot(frequencies, np.log2(frequencies))
        entropy = np.log2(samples) - total / samples  # slightly different approach to traditional method
        return 0.0 if entropy < tolerance else entropy

    @staticmethod
    def _split_data(data: np.array, split_column: np.array, split_value: float) -> Tuple[np.array, np.array]:
        """
        Splits the data on whether the column follows or deviates from the split rule.

        Parameters:
            data (np.array): A numpy array of data.
            split_column (np.array): A numpy array with the same length as the data.
            split_value (float): Used to find the values in column that are smaller than or equal to it.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple of numpy arrays. The first element is the data that follows the split
            rule. The second element is the data that does not follow the split rule.
        """

        follows_split_value = split_column <= split_value
        deviate_split_value = ~follows_split_value
        return data[follows_split_value], data[deviate_split_value]

    def _information_gain(self, X: pd.Series, y: pd.Series, split_value: float) -> float:
        """
        Calculates the information gain by splitting a feature vector based on a given split point.

        Parameters:
            X (pd.Series): Feature values.
            y (pd.Series): Class labels.
            split_value (float): The value at which to split the feature_vector.

        Returns:
            information_gain (float): The information gain obtained by splitting the feature_vector at the given split
            point.
        """

        parent_entropy = self._entropy(y)

        true_path, false_path = self._split_data(y, X, split_value)

        true_path_entropy = self._entropy(true_path)
        false_path_entropy = self._entropy(false_path)
        true_path_frac = len(true_path) / len(y)
        false_path_frac = len(false_path) / len(y)

        split_entropy = true_path_frac * true_path_entropy + false_path_frac * false_path_entropy

        information_gain = parent_entropy - split_entropy
        return information_gain

    def _best_split_value(self, X: pd.Series, y: pd.Series) -> dict:
        """
        Finds the best split point and its corresponding information gain for a given feature vector and prediction
        vector.

        Parameters:
            X (pd.Series): Feature values.
            y (pd.Series): Class labels.

        Returns:
            dict: Contains the best split point and its corresponding information gain.
            The split point is a float and the information gain is a float value between 0 and 1. If no split point
            produces a positive information gain, the function returns (None, 0.0).
        """

        best_split_value = best_information_gain = -np.inf

        split_values = map(lambda x: (x[0] + x[1]) / 2, pairwise(np.sort(np.unique(X))))
        # note: there exists alternative, more efficient split_values, if exact mid_point splits is not important
        for split_value in split_values:
            information_gain = self._information_gain(X, y, split_value)
            if information_gain >= best_information_gain:
                best_split_value = split_value
                best_information_gain = information_gain

        return {"split_value": best_split_value, "information_gain": best_information_gain}

    @staticmethod
    def _construct_leaf(y: pd.Series, node: Node) -> None:
        """
        Constructs a leaf node in the decision tree based on the class labels.

        Parameters:
            y (pd.Series): Class labels.
            node (Node): The node to be constructed as a leaf.
        """

        samples = len(y)
        if samples <= 0:
            return

        frequencies = Counter(y)
        confidences = {target: count / samples for target, count in frequencies.items()}
        node.leaf_val = max(confidences, key=confidences.get)
        node.confidences = confidences

    def _train(self, X: pd.DataFrame, y: pd.Series, stopping_depth: Optional[int] = None,
               minimum_samples: int = 0, minimum_information_gain: float = 0.0, depth: int = 0) -> Node:
        """
        Recursively builds a decision tree.

        Parameters:
            X (pd.DataFrame): List of features.
            y (pd.Series): Class labels.
            stopping_depth (int, optional): Maximum depth of the decision tree.
            minimum_samples (int): Minimum number of samples required to split a node.
            minimum_information_gain (float): Minimum information gain required to split a node.
            depth (int): Current depth of the tree during recursion.

        Returns:
            Node: The root node of the trained decision tree.
        """

        node = Node()
        node.entropy = self._entropy(y)
        node.samples = len(y)

        if node.samples <= minimum_samples:  # stopping criteria: not enough samples
            self._construct_leaf(y, node)
            return node

        if node.entropy <= 0.0:  # stopping criteria: pure leaf
            self._construct_leaf(y, node)
            return node

        if stopping_depth is not None and depth >= stopping_depth:  # stopping criteria: max depth reached
            self._construct_leaf(y, node)
            return node

        best_information_gain = 0.0
        for feature in X.columns:
            best_split = self._best_split_value(X[feature], y)
            split_value = best_split["split_value"]
            information_gain = best_split["information_gain"]
            if information_gain > 0.0 and information_gain >= best_information_gain:  # prioritizes most recent ig
                node.feature = feature
                node.split_value = split_value
                best_information_gain = information_gain

        if best_information_gain <= minimum_information_gain:  # stopping criteria: not enough information gain
            self._construct_leaf(y, node)
            return node

        true_X, false_X = self._split_data(X, X[node.feature], node.split_value)
        true_y, false_y = self._split_data(y, X[node.feature], node.split_value)
        node.left = self._train(true_X, true_y, stopping_depth, minimum_samples, minimum_information_gain, depth + 1)
        node.right = self._train(false_X, false_y, stopping_depth, minimum_samples, minimum_information_gain, depth + 1)
        return node

    def train(self, X: pd.DataFrame, y: pd.Series, stopping_depth: Optional[int] = None,
              minimum_samples: int = 0, minimum_information_gain: float = 0.0) -> Node:
        """
        Trains a decision tree classifier.

        Parameters:
            X (pd.DataFrame): List of features.
            y (pd.Series): Class labels.
            stopping_depth (int, optional): Maximum depth of the decision tree.
            minimum_samples (int): Minimum number of samples required to split a node.
            minimum_information_gain (float): Minimum information gain required to split a node.

        Returns:
            Node: The root node of the trained decision tree.
        """

        return self._train(X, y, stopping_depth, minimum_samples, minimum_information_gain)

    @staticmethod
    def display_tree(decision_tree: Node, filename: str = "tree", class_names: dict = None,
                     root_colour: str = "#eb4034", leaf_colour: str = "#50ff50", pop_up: bool = False,
                     output_format: str = "png") -> None:
        """
        Displays the decision tree using a Digraph and outputs a .png file.

        Parameters:
            decision_tree (Node): The root node of the decision tree.
            filename (str): The name of the file to be outputted.
            class_names (dict): The dictionary containing the name map from number back to titles.
            root_colour (str): Hex colour of the root nodes.
            leaf_colour (str): Hex colour of the leaf nodes.
            pop_up (bool): Automatically load the result.
            output_format(str): File format of the resulting graph.

        """

        if decision_tree is None:
            return

        int_to_hex = lambda x: format(x, "02x")
        rgb_to_hex = lambda r, g, b: f"#{int_to_hex(r)}{int_to_hex(g)}{int_to_hex(b)}"
        hex_to_rgb = lambda hex_colour: tuple(map(lambda x: int(hex_colour[x:x + 2], 16), range(1, 6, 2)))
        T = lambda t: np.power(t, 4)
        get_unique_id = lambda x: str(id(x))

        def interpolate_colour(hex_colour1: str, hex_colour2: str, t: float) -> tuple:
            rgb1 = hex_to_rgb(hex_colour1)
            rgb2 = hex_to_rgb(hex_colour2)
            h1, s1, v1 = rgb_to_hsv(*rgb1)
            h2, s2, v2 = rgb_to_hsv(*rgb2)
            h_interp = h1 + (h2 - h1) * T(t)
            s_interp = s1 + (s2 - s1) * T(t)
            v_interp = v1 + (v2 - v1) * T(t)
            r_interp, g_interp, b_interp = map(int, hsv_to_rgb(h_interp, s_interp, v_interp))
            return r_interp, g_interp, b_interp

        def tree_to_dot(node: Node, depth: int = 0) -> int:
            """
            Builds and colours the decision tree graph.

            Parameters:
                node (Node): A node of the decision tree.
                depth (int): The current depth of the node in the tree.

            Returns:
                int: The maximum depth of left and right subtrees.
            """

            left_max_depth = right_max_depth = depth

            if node.left is not None:
                left_max_depth = tree_to_dot(node.left, depth + 1)
                dot.edge(tail_name=get_unique_id(node), head_name=get_unique_id(node.left), label="T")

            if node.right is not None:
                right_max_depth = tree_to_dot(node.right, depth + 1)
                dot.edge(tail_name=get_unique_id(node), head_name=get_unique_id(node.right), label="F")

            longest_max_depth = max(left_max_depth, right_max_depth)
            t = depth / longest_max_depth if longest_max_depth != 0 else 1
            colour_interp = interpolate_colour(root_colour, leaf_colour, t)
            fill_colour = rgb_to_hex(*colour_interp)
            font_colour = "#FFFFFF" if np.average(colour_interp) <= 128 else "#000000"

            content = str(node)
            if class_names is not None:
                tmp = node.leaf_val
                node.leaf_val = class_names.get(node.leaf_val, "")
                content = str(node)
                node.leaf_val = tmp

            dot.node(name=get_unique_id(node), label=content, style="filled",
                     fillcolor=fill_colour, fontcolor=font_colour, fontname="Helvetica-Bold",
                     width="2", height="1")
            return longest_max_depth

        dot = Digraph(comment='Decision Tree')
        tree_to_dot(decision_tree)
        dot.node_attr["shape"] = "box"
        dot.render(filename, view=pop_up, format=output_format, cleanup=True)

    @staticmethod
    def train_test_split(X: pd.DataFrame, y: pd.Series, train_frac: float = 0.8, random_state: Optional[int] = None) \
            -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits the input data into training and testing sets.

        Parameters:
            X (pd.DataFrame): List of features.
            y (pd.Series): Class labels.
            train_frac (float): The fraction of the data to use for training (default 0.8).
            random_state (int): The random seed to use (default None).

        Returns:
            Tuple: Training a test data, split from original data.
        """

        if random_state is not None:
            np.random.seed(random_state)
        indices_shuffled = np.random.permutation(len(X))
        X_shuffled, y_shuffled = X.iloc[indices_shuffled], y.iloc[indices_shuffled]
        n_test = round((1 - train_frac) * len(X_shuffled))  # sketchy
        X_train, X_test = X_shuffled[n_test:], X_shuffled[:n_test]
        y_train, y_test = y_shuffled[n_test:], y_shuffled[:n_test]
        return X_train, X_test, y_train, y_test
