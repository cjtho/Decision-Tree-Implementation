# default tree packages
import numpy as np
import pandas as pd
# quality of life packages
from itertools import pairwise
from typing import Tuple, List, Optional, Generator
# displays a nice graph
from matplotlib import pyplot as plt

from node import Node
from decision_tree import DecisionTree


class DecisionTreeEvaluator:

    def __init__(self, curve: str = "ROC"):
        self.curve = curve
        if self.curve == "P-R":
            self.metric_x = "recall"
            self.metric_y = "precision"
        else:
            self.metric_x = "fpr"
            self.metric_y = "tpr"

    @staticmethod
    def accuracy(y_actual: pd.Series, y_predict: pd.Series) -> float:
        """
        Compares the predicted y values this to the actual y values for an accuracy score.

        Parameters:
            y_actual (pd.Series): The actual y values.
            y_predict (pd.Series): Predicted y values.

        Returns:
            float: Accuracy; defined as the proportion of correct predictions to total predictions.
        """

        total_predictions = len(y_actual)
        if total_predictions <= 0:
            return 0.0

        correct_predictions = (y_actual.values == y_predict.values).astype(int).sum()
        accuracy = correct_predictions / total_predictions
        return accuracy


    def test(self, decision_tree: Node, X: pd.DataFrame) -> pd.Series:
        """
        Predicts the class labels for the test data using the decision tree.

        Parameters:
            decision_tree (Node): The root of the decision tree to use for prediction.
            X (pd.DataFrame): The test set to predict class labels for.

        Returns:
            pd.Series: A pandas series with predicted class labels for the test set.
        """

        predictions = self._predict(decision_tree, X).apply(lambda node: node.leaf_val)
        return predictions

    @staticmethod
    def _performance_metrics(y_actual: pd.Series, y_probabilities: pd.Series, threshold: float) -> dict:
        """
        Calculate the performance metrics for a binary classification problem given a probability threshold.

        Parameters:
            y_actual: The actual class labels for the samples.
            y_probabilities: The predicted probabilities for the positive class.
            threshold: The probability threshold to use for classification.

        Returns:
            dict: Performance metric dictionary.
        """

        vars_to_dict = lambda **kwargs: {var_name: var_value for var_name, var_value in kwargs.items()}

        y_predict = (y_probabilities >= threshold).astype(int)  # if prob > thresh, it is positive Class
        tp = sum((y_actual == 1) & (y_predict == 1))
        fp = sum((y_actual == 0) & (y_predict == 1))
        fn = sum((y_actual == 1) & (y_predict == 0))
        tn = sum((y_actual == 0) & (y_predict == 0))

        accuracy = ((tp + tn) / (tp + tn + fp + fn)) if tp + tn + fp + fn != 0 else 0.0
        precision = (tp / (tp + fp)) if tp + fp != 0 else 1.0
        recall = tpr = sensitivity = (tp / (tp + fn)) if tp + fn != 0 else 0.0  # recall, tpr, sensitivity
        specificity = (tn / (tn + fp)) if tn + fp != 0 else 0.0
        fpr = 1 - specificity
        f1 = (2 * precision * tpr) / (precision + tpr) if precision + tpr != 0 else 0

        metrics = vars_to_dict(tp=tp, fp=fp, fn=fn, tn=tn, accuracy=accuracy, precision=precision, recall=recall,
                               tpr=tpr, sensitivity=sensitivity, specificity=specificity, fpr=fpr, f1=f1,
                               threshold=threshold)
        return metrics

    def _curve(self, y_actual: pd.Series, y_probabilities: pd.Series, threshold_step_size: float = 0.1) -> np.array:
        """
        Calculate the {self.curve} curve for a binary classification problem.

        Parameters:
            y_actual: The actual class labels for the samples.
            y_probabilities: The predicted probabilities for the true class.
            threshold_step_size (float): The step size to use when iterating through different probability thresholds.


        Returns:
            np.array: Containing the x-axis values, y-axis values, and threshold labels for each point on the
            {self.curve} curve.
        """

        thresholds = [round(x, 3) for x in np.arange(0, 1 + 2 * threshold_step_size, threshold_step_size)]
        curve = np.empty([len(thresholds), 3])
        for i, threshold in enumerate(thresholds):
            confusion_matrix = self._performance_metrics(y_actual, y_probabilities, threshold)
            x = confusion_matrix.get(self.metric_x, 0.0)
            y = confusion_matrix.get(self.metric_y, 0.0)
            curve[i] = [x, y, threshold]
        return curve

    @staticmethod
    def _auc(curve: np.array) -> float:
        """
        Calculate the area under the {self.curve} curve for a binary classification problem.

        Parameters:
            curve: The {self.curve} curve for the binary classification problem.

        Returns:
            A float representing the area under the {self.curve} curve.
        """

        curve = list(np.delete(curve, 2, axis=1))  # remove label
        curve.sort(key=lambda p: (p[0], p[1]))  # sort by x values then by y values
        # what you get when you evaluate integral of line made by 2 points
        area_under_line = lambda p1, p2: (p2[1] + p1[1]) * (p2[0] - p1[0]) * (1 / 2)
        auc = sum(area_under_line(x, y) for x, y in pairwise(curve))
        return auc

    @staticmethod
    def _predict(decision_tree: Node, X: pd.DataFrame) -> pd.Series:
        """
        Predict the class for each sample in the given dataset using the given decision tree.

        Parameters:
            decision_tree: The root of the decision tree to use for prediction.
            X: The dataset to make predictions for.

        Returns:
            A series of predicted class labels, one for each sample in the given dataset.
        """

        def _predict_row(row: pd.DataFrame, node: Node) -> Node:
            while not node.is_leaf():
                feature_val = row.get(node.feature, 0)
                if feature_val <= node.split_value:
                    node = node.left
                else:
                    node = node.right
            return node

        predictions = X.apply(_predict_row, axis=1, args=(decision_tree,))
        return predictions

    def _evaluate(self, decision_tree: Node, X: pd.DataFrame, y: pd.Series,
                  threshold_step_size: float = 0.1) -> Tuple[float, np.array]:
        """
        Evaluate the performance of a decision tree using metrics defined at class instantiation.
        IMPORTANT: Only intended for binary classification.

        Parameters:
            decision_tree (Node): Root of decision tree.
            X (pd.DataFrame): List of features.
            y (pd.Series): Class labels.
            threshold_step_size (float): The step size to use when iterating through different probability thresholds.

        Returns:
            Tuple: A float representing the overall AUC of the decision tree trained on the data and the {self.curve}
            curve.
        """

        probabilities = self._predict(decision_tree, X).apply(lambda node: node.confidences.get(1, 0))
        curve = self._curve(y, probabilities, threshold_step_size)
        auc = self._auc(curve)
        return auc, curve

    @staticmethod
    def _k_folds(X: pd.DataFrame, y: pd.Series,
                 k_folds: int = 10, random_state: Optional[int] = None) -> \
            Generator[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series], None, None]:
        """
        Generates train-test splits for k-fold cross-validation of a dataset.

        Parameters:
            X (pd.DataFrame): List of features.
            y (pd.Series): Class labels.
            k_folds (int): The number of folds to generate for cross-validation.
            random_state (int, optional): The random seed to use when shuffling the data.

        Returns:
            Generator: Yields tuples of (X_train, X_test, y_train, y_test) for each fold of the dataset.
        """

        if random_state is not None:
            np.random.seed(random_state)
        indices_shuffled = np.random.permutation(len(X))
        step_size, remainder = divmod(len(X), k_folds)

        i = 0
        while i < len(X):
            if remainder > 0:
                step = step_size + 1
                remainder -= 1
            else:
                step = step_size
            test_indices = indices_shuffled[i:i + step]
            train_indices = np.delete(indices_shuffled, np.array(list(range(i, i + step))))
            X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
            y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
            yield X_train, X_test, y_train, y_test
            i += step


    def cross_validate(self, tree_model: DecisionTree, X: pd.DataFrame, y: pd.Series,
                       stopping_depth: Optional[int] = None, threshold_step_size: float = 0.1, k_folds: int = 10,
                       random_state: Optional[int] = None) -> dict:
        """
        Perform k-fold cross-validation of a decision tree model.

        Parameters:
            tree_model (DecisionTree): The decision tree model to be trained and evaluated.
            X (pd.DataFrame): List of features.
            y (pd.Series): Class labels.
            stopping_depth: The maximum depth of the decision tree.
            threshold_step_size (float): The step size to use when iterating through different probability thresholds.
            k_folds (int): The number of folds to generate for cross-validation.
            random_state (int, optional): The random seed to use when shuffling the data.

        Returns:
            dict: The evaluation results for each fold, including the AUC score and corresponding {self.curve} curve.
        """

        info = {"train_auc": [], f"{self.curve}_train_curves": [], "test_auc": [], f"{self.curve}_test_curves": []}
        folds = self._k_folds(X, y, k_folds, random_state)
        for X_train, X_test, y_train, y_test in folds:
            decision_tree = tree_model.train(X_train, y_train, stopping_depth)
            train_auc, train_curve = self._evaluate(decision_tree, X_train, y_train, threshold_step_size)
            test_auc, test_curve = self._evaluate(decision_tree, X_test, y_test, threshold_step_size)
            info["train_auc"].append(train_auc)
            info["test_auc"].append(test_auc)
            info[f"{self.curve}_train_curves"].append(train_curve)
            info[f"{self.curve}_test_curves"].append(test_curve)

        return info

    def display_curves(self, curves: List[np.array], show_thresholds: bool = False,
                       title: str = "") -> None:
        """
        Display {self.curve} curves for a list of models.

        Parameters:
            curves (List[np.array]): A list of numpy arrays containing the coordinates of {self.curve} curves.
            show_thresholds (bool): Decides whether points have their thresholds annotated.
            title (str): Titles the graph.
        """

        graph_colours = ["red", "green", "blue", "yellow", "orange", "purple"]

        if self.curve == "P-R":
            plt.plot([0, 1], [1, 0], "k--")
        else:
            plt.plot([0, 0], [1, 1], "k--")

        for i, curve in enumerate(curves):
            colour = graph_colours[i % len(graph_colours)]
            x_values = curve[:, 0]
            y_values = curve[:, 1]
            labels = curve[:, 2]
            plt.fill_between(x_values, y_values, alpha=0.1, color=colour)
            seen_coordinates = set()
            for x, y, label in zip(x_values, y_values, labels):
                if (x, y) in seen_coordinates:
                    continue
                seen_coordinates.add((x, y))
                plt.scatter(x, y, marker="o", color=colour)
                if show_thresholds:
                    plt.text(x, y + 0.01, "{:.2f}".format(label), ha="center")
            plt.plot(x_values, y_values, color=colour, label=f"AUC = {self._auc(curve)}")

        plt.title(f"{self.curve} Curves" if title == "" else title)
        plt.xlabel(self.metric_x)
        plt.ylabel(self.metric_y)
        plt.grid()
        plt.legend()
        plt.show()
