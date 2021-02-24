import time
from math import sqrt
from random import choice

from numpy import log


class Node:
    def __init__(self, board, depth=0, parent=None):
        """
        A node in the MC Search Tree.

        Will represent a single board state
        """

        # Score from previous simulations
        # +1 for win
        # 0 for loss
        # +0.5 for tie
        self.t = 0

        # Number of previous simulations
        self.n = 0

        # List of nodes that are this node's children
        self.children = []

        # The current game state
        self.board = board

        # How many layers deep is this simulation
        self.depth = depth

        # The parent node
        self.parent = parent

        if depth > 0 and parent is None:
            raise RuntimeError("Node defined with depth>1 and parent of None")

    def calc_UCB_of_child(self, child, final_score=False):
        """
        Calculates the UCB of a child node with self as the parent.

        UCB is defined by the following formula:

        UCB1 = Vi + 2 * sqrt( ln(n) / ni )

        Where,
        Vi is average reward of all nodes beneath this node (the child) (calculated by t/n)
        N is the number of times the parent has been visited
        ni is the number of times the child node has been visited

        This function should not be called before a rollout is done on the parent (ie self.n = 0)

        :param child: Node, the child
        :param final_score: bool, If true, childen with n=0 with return -inf; If false, children with n=0 will return inf
        :return: float, the UCB value
        """

        # Prevent divide by zero errors
        if child.n <= 0:

            if final_score:
                return float("-inf")
            else:
                return float("inf")

        # Parent must have a rollout before calling this function
        if self.n <= 0:
            raise RuntimeError(
                "Cannot calculate UCB of child when parent node has N of 0"
            )

        return float((child.t / child.n) + 2 * sqrt(log(self.n) / child.n))

    def rollout_random(self, play_x=True, verbose=False):
        """
        Calculates the result of the game from this position with both players playing random moves.

        :param play_x: True to find optimal move for X, False to find optimal move for O. (default=True)
        :param verbose: True to print out every single board, False (default) to not
        :return: float, 1 for x win, 0 for O win, and 0.5 for tie
        """

        temp_board = self.board.copy_board()

        # Pick random moves until the game is over
        while temp_board is None:
            temp_board = choice(temp_board.all_possible_moves())

            if verbose:
                print(temp_board)

        # Return the appropriate value based on the result
        result = temp_board.game_result

        # If win
        if (result == "X" and play_x) or (result == "O" and not play_x):
            return 1

        # If lose
        elif (result == "O" and play_x) or (result == "X" and not play_x):
            return 0

        # If tie
        elif not result:
            return 0.5

        else:
            raise RuntimeError(f"Unexpected game result: {result}")

    @property
    def is_terminal(self):
        """
        Is this node terminal
        :return: boolean, True if terminal, False if not
        """
        # TODO: Optimization, once this result has been calculated once, store it and do not recalculate
        result = self.board.game_result

        # If the result is not none, the node is terminal
        if result is not None:
            return True

        return False

    def add_children(self):
        """
        Calculates all possible moves and adds them as child nodes.
        :return: True if children are added, False if children cannot be added (ie terminal state)
        """

        # Warn if children have already been added
        if len(self.children) > 0:
            raise RuntimeWarning(
                f"Attempting to add children to a node with {len(self.children)} children."
            )

        # If the game is still going, add more children
        if not self.is_terminal:
            for i in self.board.all_possible_moves():
                # Each child has a depth of one greater than the parent
                self.children.append(Node(i, self.depth + 1, self))

            return True

        return False

    @property
    def ucb(self):
        return self.parent.calc_UCB_of_child(self)

    @property
    def final_score(self):
        return self.calc_UCB_of_child(self, True)

    def backpropagate(self, result):
        """
        Backpropagates the result of a rollout to all parent nodes recursively

        :param result: float, the result for backpropagation
        :return: None
        """

        # There has been one more simulation carried out
        self.n += 1

        # The result is added
        self.t += result

        # Backpropagate to the parent, if one exists
        if self.parent is not None:
            self.parent.backpropagate(result)

    def __str__(self):
        return f"Node\nDepth: {self.depth}\nIs terminal: {self.is_terminal}\nNum Children: {len(self.children)}\nN: {self.n}\nT: {self.t}\n{self.board}"

    def descendants_recursive(self):
        current_count = len(self.children)

        for i in self.children:
            current_count += i.descendants_recursive()

        return current_count

    @property
    def descendants(self):
        """
        Calculates the number of nodes below self on the tree recursively
        :return: int, the number of nodes
        """
        return self.descendants_recursive()


def mcts(
    start_node: Node, iterations: int, think_time: int = None, verbose: bool = False
) -> Node:
    """
    Performs a Monte Carlo Tree Search for the given number of iterations.

    :param start_node: Node, the starting node / initial state
    :param iterations: int, number of iterations to be performed
    :param think_time: int, number of seconds to think for (default=None), will be prioritized over iterations if set
    :param verbose: bool, True to print out all nodes and simulation results, (default=False)
    :return: Node, the node with the highest score (ie the recommended move)
    """

    print(start_node.board)

    def main_loop():
        if verbose:
            print(f"Starting iteration {_}")

        # Start back at the beginning for every iteration
        current_node = start_node

        # Loop until a rollout is completed
        rollout_complete = False
        while not rollout_complete:

            # STEP 1: Selection
            # Find the best child node to roll out from

            # Check if current node is a leaf node (ie does it have children)
            if len(current_node.children) > 0:
                # Not a leaf node

                # Function that calculates the UCB of the given node

                # Select the child node with the greatest UCB1 score as the next node to explore
                current_node = max(current_node.children, key=lambda child: child.ucb)

            # Current node is a leaf node
            else:

                # If previous simulations have been done, add children and select a new one
                # If no simulations have been done, rollout from here
                if current_node.n > 0:
                    children_added = current_node.add_children()

                    # If there are new nodes added
                    if children_added:
                        # There are freshly created nodes, so they all will have identical UCBs
                        # Just select a random node for rollout
                        current_node = choice(current_node.children)

                    # If no new nodes are added (current_node is terminal) just rollout from current_node

                rollout_result = current_node.rollout_random()
                rollout_complete = True

                current_node.backpropagate(rollout_result)

    if think_time is None:
        for _ in range(iterations):
            main_loop()

    else:
        start_time = time.time()
        iterations_performed = 0
        while time.time() < start_time + think_time:
            iterations_performed += 1
            main_loop()

        print(f"{iterations_performed} iterations performed")

    # After all iterations are done, return the child node with this highest UCB1
    return max(start_node.children, key=lambda child: child.final_score)
