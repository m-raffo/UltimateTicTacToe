import time
from math import sqrt
from random import choice

from numpy import log


def flip_board(board):
    """
    Flips the X and O pieces on a large scale board.
    :param board: The board
    :return: list, the converted board
    """

    flipped_board = []

    for i in board:
        flipped_board.append([])
        # Flip X and O in every list
        # I is just an intermediary character, no other significance
        for spot in i:
            if spot == "O":
                flipped_board[-1].append("X")
            elif spot == "X":
                flipped_board[-1].append("O")
            else:
                flipped_board[-1].append(None)

    return flipped_board


def mini_board_eval(miniboard: list) -> float:
    """
    Calculates the evaluation of a mini board.

    eval = (c1 * w) + (c2 * r)

    Where:
    c1 and c2 = constants
    w = spaces that are winning if taken
    r = rows with one spot taken (ie win in two moves)

    :param miniboard: The miniboard to evaluate
    :return: The evaluation for X
    """

    # Mini board eval constants
    c1 = 2
    c2 = 1

    # Evals of won and lost boards
    cw = 10
    cl = 0

    # This will be incremented as matching positions are found
    r = 0

    winning_index = set()

    # Check each row
    for i in range(3):
        row = miniboard[i * 3 : i * 3 + 3]
        set_row = set(row)

        # Check for win
        if set_row == {"X"}:
            return cw

        # Check for loss
        if set_row == {"O"}:
            return cl

        # If row is empty except for x, this is a row winnable in two moves
        if set_row == {"X", None}:
            count = row.count("X")
            if count == 1:
                r += 1

            # If the row is empty except for one, this is a winning index
            elif count == 2:
                winning_index.add(i)

    # Check each column
    for i in range(3):
        row = [miniboard[i], miniboard[i + 3], miniboard[i + 6]]
        set_row = set(row)

        # Check for win
        if set_row == {"X"}:
            return cw

        # Check for loss
        if set_row == {"O"}:
            return cl

        # If row is empty except for x, this is a row winnable in two moves
        if set_row == {"X", None}:
            count = row.count("X")
            if count == 1:
                r += 1

            # If the row is empty except for one, this is a winning index
            elif count == 2:
                winning_index.add(i)

    # Check both diagonals
    for row in [
        [miniboard[0], miniboard[4], miniboard[8]],
        [miniboard[2], miniboard[4], miniboard[6]],
    ]:
        set_row = set(row)

        # Check for win
        if set_row == {"X"}:
            return cw

        # Check for loss
        if set_row == {"O"}:
            return cl

        # If row is empty except for x, this is a row winnable in two moves
        if set_row == {"X", None}:
            count = row.count("X")
            if count == 1:
                r += 1

            # If the row is empty except for one, this is a winning index
            elif count == 2:
                winning_index.add(i)

    w = len(winning_index)

    return c1 * w + c2 * r


def calc_significance(board):
    """
    Calculates the significance of each miniboard.

    Significance = sum(eval of all boards this board could be involved in a win with)

    :return: list[float] The significance of each board
    """

    # Calculate each mini board evaluation
    evals = []
    for i in board:
        evals.append(mini_board_eval(i))

    # All win patterns (Note each list does not include the board itself)
    win_possibilities = {
        0: [[1, 2], [4, 8], [3, 6]],
        1: [[4, 7], [0, 2]],
        2: [[4, 6], [0, 1], [5, 8]],
        3: [[0, 6], [4, 5]],
        4: [[0, 8], [1, 7], [2, 6], [3, 5]],
        5: [[3, 4], [2, 8]],
        6: [[0, 3], [7, 8], [4, 2]],
        7: [[6, 8], [1, 4]],
        8: [[0, 4], [6, 7], [2, 5]],
    }

    # Calculate the significance of each board (default is 1)
    significances = []
    for i in range(9):
        significances.append(1)
        # Check each win possibility
        # TODO: Check if the win is possible before adding the evals?
        for win_coordinates in win_possibilities[i]:

            miniboard1_eval = evals[win_coordinates[0]]
            miniboard2_eval = evals[win_coordinates[1]]

            significances[-1] += miniboard1_eval + miniboard2_eval

    return significances


def eval_board_one_side(board):
    """
    Calculate an evaluation of a full board for one side
    :param board: list the board
    :return: float - A number that indicates that extent to which X is winning
    """

    miniboard_evals = []

    for miniboard in board:
        miniboard_evals.append(mini_board_eval(miniboard))

    significances = calc_significance(board)

    final_eval = 0

    for i in range(9):
        final_eval += miniboard_evals[i] * significances[i]

    return final_eval


def eval_board(board):
    """
    Calculate a full evaluation for both sides of a large board.
    :param board:
    :return: float - positive indicates that X is winning; negative indicates O is winning
    """

    x_eval = eval_board_one_side(board)
    flipped = flip_board(board)
    o_eval = eval_board_one_side(flipped)

    return x_eval - o_eval


def detail_eval(board):
    x_eval = eval_board_one_side(board)
    flipped = flip_board(board)
    o_eval = eval_board_one_side(flipped)

    return x_eval, o_eval


def minimax(board, depth: int):
    """
    Calculates the best move based on minimax evaluation of the given depth.
    :param board: The board to evaluate from
    :param depth: The number of moves to search into the future
    :return: [board, move] The ideal move for X to make
    """

    root_node = board
    children = [root_node]

    # Add all the children
    for i in range(depth):
        print(f"Depth: {i}")
        new_children = []
        for child in children:
            # Get next moves
            if len(child.children) == 0:
                child.add_children()
            # Add all the children to be expanded on next
            new_children.extend(child.children)

        # Next time loop through all the newly added children
        children = new_children[:]
        print(len(children))

    print("Searching best move...")
    # Find the possible move with the highest evaluation
    best_move = max(root_node.children, key=lambda x: x.calc_eval_from_children(True))

    return best_move


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

        # Approx the square root of 2
        c = 1.4142
        return float((child.t / child.n) + c * sqrt(log(self.n) / child.n))

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
            return -1

        # If tie
        elif not result:
            return 0

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

    def calc_eval_from_children(self, get_min=False):
        """
        Calculates the evaluation from its children's evaluations, recursively.
        If there are no children, return its heuristic evaluation.
        :param get_min: Set to true to get the lowest child evaluation
        :return: float; the evaluation based on children
        """

        # If no children, return heuristic evaluation
        if len(self.children) == 0:
            return self.eval

        # Get all the evaluations
        values = []
        for i in self.children:
            values.append(i.calc_eval_from_children(not get_min))

        # Flip get_min with each turn
        if get_min:
            return min(values)

        return max(values)

    @property
    def ucb(self):
        return self.parent.calc_UCB_of_child(self)

    @property
    def final_score(self):
        return self.calc_UCB_of_child(self, True)

    @property
    def eval(self):
        """
        Calculates the boards evaluation
        :return: float; the evaluation
        """

        # If the game is over, return appropriate value
        result = self.board.game_result
        if result == "X":
            return float("inf")
        elif result == "O":
            return float("-inf")
        # If tie return 0
        elif result == False:
            return 0

        return eval_board(self.board.board)

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
    start_node: Node,
    iterations: int,
    think_time: int = None,
    adjust_think_time=False,
    verbose: bool = False,
) -> Node:
    """
    Performs a Monte Carlo Tree Search for the given number of iterations.

    :param adjust_think_time:
    :param start_node: Node, the starting node / initial state
    :param iterations: int, number of iterations to be performed
    :param think_time: int, number of seconds to think for (default=None), will be prioritized over iterations if set
    :param adjust_think_time: bool, set to True to allow the for more iterations when there are more available moves (default=False)
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

        if len(start_node.board.all_possible_moves()) > 9:
            new_think_time = think_time * 2
        else:
            new_think_time = think_time

        while time.time() < start_time + new_think_time:
            iterations_performed += 1
            main_loop()

        print(f"{iterations_performed} iterations performed")

    # After all iterations are done, return the child node with this highest UCB1
    return max(start_node.children, key=lambda child: child.final_score)
