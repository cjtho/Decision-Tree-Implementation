class Node:
    def __init__(self):
        self.feature = None
        self.split_value = None
        self.entropy = None
        self.samples = None
        self.leaf_val = None
        self.confidences = dict()
        self.left = None
        self.right = None

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def __str__(self) -> str:
        lines = []
        # branch nodes
        if self.feature is not None:
            lines.append(f"{self.feature} <= {self.split_value}")

        # all nodes
        if self.entropy is not None:
            lines.append(f"entropy = {self.entropy}" if self.entropy.is_integer()
                         else f"entropy = {self.entropy:.3f}")
        if self.samples is not None:
            lines.append(f"samples = {self.samples}")

        # leaf nodes
        if self.is_leaf():
            lines.append(f"class = {self.leaf_val}")
            leaf_confidence = max(self.confidences.values(), default=0.0)
            lines.append(f"confidence = {leaf_confidence}" if leaf_confidence.is_integer()
                         else f"confidence = {leaf_confidence:.3f}")
        return "\n".join(lines)