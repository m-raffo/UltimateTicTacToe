from cmath import sqrt
from random import choice

from numpy import log


class Node:
    def __init__(self, board):
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

    def calc_UCB_of_child(self, child):
        """
        Calculates the UCB of a child node with self as the parent.

        UCB is defined by the following formula:

        UCB1 = Vi + 2 * sqrt( ln(n) / ni )

        Where,
        Vi is average reward of all nodes beneath this node (the child) (calculated by t/n)
        N is the number of times the parent has been visited
        ni is the number of times the child node has been visited

        :param child: Node, the child
        :return: float, the UCB value
        """

        return float((child.t / child.n) + 2 * sqrt(log(self.n) / child.n))

    def rollout_random(self, verbose=False):
        """
        Calculates the result of the game from this position with both players playing random moves.
        :return: int, 1 for x win, 0 for O win, and 0.5 for tie
        """

        # Pick random moves until the game is over
        while self.board.game_result is None:
            self.board = choice(self.board.all_possible_moves())

            if verbose:
                print(self.board)

        # Return the appropriate value based on the result
        result = self.board.game_result
        if result == "X":
            return 1
        elif result == "O":
            return 0
        elif not result:
            return 0.5
        else:
            raise RuntimeError(f"Unexpected game result: {result}")
