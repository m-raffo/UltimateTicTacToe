import time
from functools import partial
from math import sqrt
from random import choice
from multiprocessing import Pool
from numpy import log
import numpy as np
cimport numpy as np
from cython.view cimport array as cvarray

# NOTE:
# All X's have been replaced with 1
# ALL O's have been replaced with -1
# All Empty Spaces/None have been replaced with 0
# This is to allow for greater optimization in cython and numpy arrays


cdef int[:] all_x = np.full(3, 1, dtype=np.int32)
cdef int[:] all_o = np.full(3, -1, dtype=np.int32)
cdef int[:] all_empty = np.full(3, 0, dtype=np.int32)
cdef int[:] index_order = np.array([0, 4, 8, 2, 4, 6], dtype=np.int32)
cdef int[:, :] empty_board = np.full((9, 9), 0, dtype=np.int32)
cdef int[:] empty_miniboard = np.full(9, 0, dtype=np.int32)


cpdef int [:,:] flip_board(int [:, :] board):
    """
    Flips the X and O pieces on a large scale board.
    :param board: The board
    :return: list, the converted board
    """

    cdef int flipped_board_array[9][9]
    cdef int[:, :] flipped_board = flipped_board_array


    cdef int spot

    for i in range(board.shape[0]):

        # Flip X and O in every list
        # I is just an intermediary character, no other significance

        for j in range(board.shape[1]):
            flipped_board[i, j] = board[i, j] * -1

    return flipped_board


cdef int count_instances(int [:] row_to_count, int value):
    """
    A version of list.count() for memoryviewslices
    """
    cdef int result = 0
    for i in range(row_to_count.shape[0]):
        if row_to_count[i] == value:
          result += 1

    return result

cdef int find_index(int [:] row_to_search, int value):
    for i in range(row_to_search.shape[0]):
        if row_to_search[i] == value:
          return i

    return -1

cdef int check_equality(int[:] array1, int[:] array2):
    if array1.shape[0] != array2.shape[0]:
        return 0

    for i in range(array1.shape[0]):
        if array1[i] != array2[i]:
          return 0

    return 1

cpdef float mini_board_eval(int [:] miniboard, constants=None):
    """
    Calculates the evaluation of a mini board.

    eval = (c1 * (w ^ 0.5)) + (c2 * r)

    Where:
    c1 and c2 = constants
    w = spaces that are winning if taken
    r = rows with one spot taken (ie win in two moves)

    Constants are:
    c1: the value of a space that is winning if taken
    c2: the value of a row that can win in two moves
    cw: the value of an already won board
    cl: the value of a lost board

    :param constants: Optional, custom constant values to use to evaluate the board
    :param miniboard: The miniboard to evaluate
    :return: The evaluation for X
    """

    cdef int c1, c2, cw, cl

    if constants is None:
        # Mini board eval constants
        c1 = 2
        c2 = 1

        # Evals of won and lost boards
        cw = 10
        cl = 0

    # Use custom constants if specified
    else:
        c1 = constants["c1"]
        c2 = constants["c2"]

        cw = constants["cw"]
        cl = constants["cl"]

    # This will be incremented as matching positions are found
    cdef int r = 0

    winning_index = set()



    cdef int[:] row

    # Check each row
    cdef int i
    for i in range(3):
        row = miniboard[i * 3 : i * 3 + 3]

        # Check for win
        if check_equality(all_x, row):
            return cw

        # Check for loss
        if check_equality(all_o, row):
            return cl

        # If row is empty except for x, this is a row winnable in two moves
        if -1 not in row:
            count = count_instances(row, 1)
            if count == 1:
                r += 1

            # If the row is empty except for one, this is a winning index
            elif count == 2:
                # Calculate the location of the empty square in the whole board
                winning_index.add(i * 3 + find_index(row, 0))

    # Check each column
    for i in range(3):
        row = miniboard[i:i+7:3]

        # Check for win
        if check_equality(all_x, row):
            return cw

        # Check for loss
        if check_equality(all_o, row):
            return cl

        # If row is empty except for x, this is a row winnable in two moves
        if -1 not in row:
            count = count_instances(row, 1)
            if count == 1:
                r += 1

            # If the row is empty except for one, this is a winning index
            elif count == 2:
                # Calculate the location of the empty square in the whole board
                winning_index.add(i + find_index(row, 0) * 3)

    # Check both diagonals

    # Keep track of which space is which on the miniboard
    cdef int diagonal_count = -1

    cdef int[:, :] diagonals = np.array([
        [miniboard[0], miniboard[4], miniboard[8]],
        [miniboard[2], miniboard[4], miniboard[6]],
    ], dtype=np.int32)

    for row_index in range(2):
        row = diagonals[row_index]
        diagonal_count += 1

        # Check for win
        if check_equality(all_x, row):
            return cw

        # Check for loss
        if check_equality(all_o, row):
            return cl

        # If row is empty except for x, this is a row winnable in two moves
        if -1 not in row:
            count = count_instances(row, 1)
            if count == 1:
                r += 1

            # If the row is empty except for one, this is a winning index
            elif count == 2:
                # Add the correct index
                empty_space_index = index_order[find_index(row, 0) + 3 * diagonal_count]
                winning_index.add(empty_space_index)

    cdef int w = len(winning_index)

    return c1 * (w ** 0.5) + c2 * r

cdef int _check_rows_and_columns_and_diagonals(int [:] b):
    # Check each row
    cdef int i
    cdef int [:] row
    cdef int row_index


    for i in range(3):
        row = b[i * 3 : i * 3 + 3]

        # Check for win
        if check_equality(row, all_x):
            return 1

        # Check for loss
        if check_equality(row, all_o):
            return -1

    # Check each column
    for i in range(3):
        row = b[i:i+7:3]

        # Check for win
        if check_equality(row, all_x):
            return 1

        # Check for loss
        if check_equality(row, all_o):
            return -1

    cdef int[:, :] diagonals = np.array([
        [b[0], b[4], b[8]],
        [b[2], b[4], b[6]],
    ], dtype=np.int32)

    # Check both diagonals
    for row_index in range(2):
        row = diagonals[row_index]

        # Check for win
        if check_equality(row, all_x):
            return 1

        # Check for loss
        if check_equality(row, all_o):
            return -1

    return 0

cpdef int check_win(int[:] board):
    """
    Checks if the given board is a win (ie three in a row)
    :param board: A list of the board
    :return: 1 if x is winning, -1 if o is winning, 2 if neither is winning and the game goes on, 0 if a tie
    """


    # If the board is empty, the game goes on
    if check_equality(board, empty_miniboard):
        return 2

    cdef int result = _check_rows_and_columns_and_diagonals(board)

    if result != 0:
        return result

    # If the board is filled, tie; if not, the game continues
    if 0 in board:
        return 2

    return 0

# All win patterns (Note each list does not include the board itself)
# [-1, -1] Included to make the array the same length in all elements
cdef int[:, :, :] win_possibilities = np.array([
    [[1, 2], [4, 8], [3, 6], [-1, -1]],
    [[4, 7], [0, 2], [-1, -1], [-1, -1]],
    [[4, 6], [0, 1], [5, 8], [-1, -1]],
    [[0, 6], [4, 5], [-1, -1], [-1, -1]],
    [[0, 8], [1, 7], [2, 6], [3, 5]],
    [[3, 4], [2, 8], [-1, -1], [-1, -1]],
    [[0, 3], [7, 8], [4, 2], [-1, -1]],
    [[6, 8], [1, 4], [-1, -1], [-1, -1]],
    [[0, 4], [6, 7], [2, 5], [-1, -1]],
], dtype=np.int32)

cdef float[:] calc_significance(int [:, :] board):
    """
    Calculates the significance of each miniboard.

    Significance = sum(eval of all boards this board could be involved in a win with)

    :return: list[float] The significance of each board
    """

    # Calculate each mini board evaluation
    cdef float[9] evals
    cdef int i
    cdef float miniboard1_eval, miniboard2_eval

    for i in range(board.shape[0]):
        evals[i] = mini_board_eval(board[i])

    # Calculate the significance of each board (default is 1)
    cdef float[9] significances
    cdef int[:] win_coordinates

    cdef int game_results0, game_results1

    for i in range(9):
        significances[i] = 1


        # Check each win possibility
        for win_coordinates in win_possibilities[i]:
            # If this is true, they have all been serached
            if win_coordinates[0] == -1:
                break
            # If either board is already won for the other side, a win is not possible
            # or if either board is already drawn
            game_results0 = check_win(board[win_coordinates[0]])
            game_results1 = check_win(board[win_coordinates[1]])
            if (
                game_results0 == 0
                or game_results0 == -1
                or game_results1 == 0
                or game_results1 == -1
            ):
                continue

            miniboard1_eval = evals[win_coordinates[0]]
            miniboard2_eval = evals[win_coordinates[1]]

            significances[i] += miniboard1_eval + miniboard2_eval

    return significances


cdef float eval_board_one_side(int[:, :]board, constants=None):
    """
    Calculate an evaluation of a full board for one side
    :param constants: Optional, the constants to be used when evaluating the position. See mini_board_eval for more information
    :param board: list the board
    :return: float - A number that indicates that extent to which X is winning
    """

    cdef float[9] miniboard_evals

    for miniboard_index in range(board.shape[0]):
        miniboard_evals[miniboard_index] = mini_board_eval(board[miniboard_index], constants)

    cdef float[:] significances = calc_significance(board)

    cdef float final_eval = 0

    cdef int i
    for i in range(9):
        final_eval += miniboard_evals[i] * significances[i]

    return final_eval


cdef float eval_board(int[:, :] board, constants=None):
    """
    Calculate a full evaluation for both sides of a large board.
    :param constants: Optional, the constants to be used when evaluating the position. See mini_board_eval for more information.
    :param board:
    :return: float - positive indicates that X is winning; negative indicates O is winning
    """

    cdef float x_eval = eval_board_one_side(board, constants)
    cdef int [:,:] flipped = flip_board(board)
    cdef float o_eval = eval_board_one_side(flipped, constants)

    return x_eval - o_eval


cpdef int[:] detail_eval(int [:, :] board):
    cdef float x_eval, o_eval
    cdef int[:, :] flipped
    cdef int[:] result

    x_eval = eval_board_one_side(board)
    flipped = flip_board(board)
    o_eval = eval_board_one_side(flipped)

    result = np.array([x_eval, o_eval], dtype=np.int32)

    return result


cpdef minimax(board, int depth, float alpha, float beta, maximizing_player, constants=None):
    """
    Calculates the best move based on minimax evaluation of the given depth.
    :param maximizing_player:
    :param beta:
    :param alpha:
    :param board: The board to evaluate from
    :param depth: The number of moves to search into the future
    :return:
    """
    if depth == 0 or board.board.game_result() != 2:
        final_evaluation = board.eval_constants(constants)

        # Save the depth if the evaluation in infinite
        if final_evaluation == float("inf") or final_evaluation == float("-inf"):
            board.inf_depth = board.depth
        return board.eval_constants(constants)

    # Initialize with worst possible outcome, so anything else is always better
    if maximizing_player:
        best_eval = float("-inf")
    else:
        best_eval = float("inf")

    # Add children if necessary
    if len(board.children) == 0:
        board.add_children()

    # Get evaluations of all children
    for child in board.children:

        # Don't evaluate pruned children
        if child.pruned:
            continue

        # Use recursion to search the tree
        new_eval = minimax(
            child, depth - 1, alpha, beta, not maximizing_player, constants
        )

        # Get next best eval and use alpha beta pruning
        if maximizing_player:
            best_eval = max(best_eval, new_eval)

            # If the position is lost, the best option is the one furthest from a win
            if best_eval == float("-inf"):
                if board.inf_depth is None:
                    board.inf_depth = child.inf_depth
                elif board.inf_depth < child.inf_depth:
                    board.inf_depth = child.inf_depth

            alpha = max(alpha, new_eval)
            if beta <= alpha:
                board.pruned = True
                break
        else:
            if new_eval == float("-inf"):
                board.inf_depth = child.inf_depth

            best_eval = min(best_eval, new_eval)

            # If the position is lost, the best option is the one furthest from a win
            if best_eval == float("inf"):
                if board.inf_depth is None:
                    board.inf_depth = child.inf_depth
                elif board.inf_depth < child.inf_depth:
                    board.inf_depth = child.inf_depth

            beta = min(beta, new_eval)
            if beta <= alpha:
                board.pruned = True
                break
    return best_eval


def minimax_search_async(board, depth, play_as_o=False, constants=None):

    if len(board.children) == 0:
        board.add_children()

    # Pruning might not be working optimally b/c the first set of child nodes are all kept separate
    minimax_partial = partial(
        minimax,
        depth=depth - 1,
        alpha=float("-inf"),
        beta=float("inf"),
        maximizing_player=play_as_o,
        constants=constants,
    )

    p = Pool()
    evals = p.map_async(minimax_partial, board.children)

    return evals, p


# Pruning might not be working optimally b/c the first set of child nodes are all kept separate
def minimax_prune(board, depth1, depth2, play_as_o=False):
    # Step 1: First round of evaluations
    minimax(board, depth1 - 1, float("-inf"), float("inf"), play_as_o)

    # Step 2: Second round of evaluations
    return minimax(board, depth2 - 1, float("-inf"), float("inf"), play_as_o)

def minimax_search_pruning(board, depth1, depth2, play_as_o=False, constants=None):
    if len(board.children) == 0:
        board.add_children()


    minimax_partial = partial(
        minimax_prune,
        depth1=depth1,
        depth2=depth2,
        play_as_o=play_as_o,
    )

    with Pool() as p:
        evals = p.map(minimax_partial, board.children)

    moves_and_evals = list(zip(board.children, evals))

    print(moves_and_evals)
    if not play_as_o:

        return max(moves_and_evals, key=lambda x: (x[1], x[0].inf_depth))
    else:
        return min(moves_and_evals, key=lambda x: (x[1], -x[0].inf_depth))


def minimax_prune_variable_depth(board, depths, play_as_o):
    for i in depths[:-1]:
        minimax(board, i - 1, float("-inf"), float("inf"), play_as_o)


    # Step 2: Second round of evaluations
    return minimax(board, depths[-1] - 1, float("-inf"), float("inf"), play_as_o)


def minimax_prune_variable_depth_time_limited(board, depths, time_limit, play_as_o):
    end_time = time.time() + time_limit

    for i in depths[:-1]:
        result = minimax(board, i - 1, float("-inf"), float("inf"), play_as_o)

        # Stop searching after the time has passed and give result
        if time.time() > end_time:
            return result

    # If time has not yet run out, do the final depth evaluation
    return minimax(board, depths[-1] - 1, float("-inf"), float("inf"), play_as_o)

def minimax_search_prune_time_limited_async(board, depths, time_limit, play_as_o=False):
    if len(board.children) == 0:
        board.add_children()

    # Pruning might not be working optimally b/c the first set of child nodes are all kept separate

    minimax_partial = partial(
        minimax_prune_variable_depth_time_limited,
        depths=depths,
        time_limit=time_limit,
        play_as_o=play_as_o,
    )

    p = Pool()
    evals = p.map_async(minimax_partial, board.children)

    return evals, p

def minimax_search_seq_variable_pruning(board, depths, play_as_o=False, constants=None):


    if len(board.children) == 0:
        board.add_children()


    for depth in depths[:-1]:
        # Step 1: First round of evaluations and pruning
        for i in board.children:
            minimax(i, depth - 1, float("-inf"), float("inf"), play_as_o)


    # Step 2: Second round of evaluations
    moves_and_evals = []
    for i in board.children:
        moves_and_evals.append(
            [i, minimax(i, depths[-1] - 1, float("-inf"), float("inf"), play_as_o)]
        )

    print(list(moves_and_evals))

    if not play_as_o:

        return max(moves_and_evals, key=lambda x: (x[1], x[0].inf_depth))
    else:
        return min(moves_and_evals, key=lambda x: (x[1], -x[0].inf_depth))


def minimax_search_variable_pruning_async(board, depths, play_as_o=False, constants=None):
    if len(board.children) == 0:
        board.add_children()

    # Pruning might not be working optimally b/c the first set of child nodes are all kept separate

    minimax_partial = partial(
        minimax_prune_variable_depth,
        depths=depths,
        play_as_o=play_as_o,
    )

    p = Pool()
    evals = p.map_async(minimax_partial, board.children)

    return evals, p

def minimax_search_pruning_async(board, depth1, depth2, play_as_o=False, constants=None):
    if len(board.children) == 0:
        board.add_children()

    # Pruning might not be working optimally b/c the first set of child nodes are all kept separate

    minimax_partial = partial(
        minimax_prune,
        depth1=depth1,
        depth2=depth2,
        play_as_o=play_as_o,
    )

    p = Pool()
    evals = p.map_async(minimax_partial, board.children)

    return evals, p


def minimax_search_seq_pruning(board, depth1, depth2, play_as_o=False, constants=None):
    """
    Search for the best move, performing one round of pruning after depth1, then processing to depth 2.
    :param board:
    :param depth:
    :param play_as_o:
    :param constants:
    :return:
    """

    if len(board.children) == 0:
        board.add_children()


    # Step 1: First round of evaluations and pruning
    for i in board.children:
        minimax(i, depth1 - 1, float("-inf"), float("inf"), play_as_o)


    # Step 2: Second round of evaluations
    moves_and_evals = []
    for i in board.children:
        moves_and_evals.append(
            [i, minimax(i, depth2 - 1, float("-inf"), float("inf"), play_as_o)]
        )

    print(list(moves_and_evals))

    if not play_as_o:

        return max(moves_and_evals, key=lambda x: (x[1], x[0].inf_depth))
    else:
        return min(moves_and_evals, key=lambda x: (x[1], -x[0].inf_depth))


def minimax_search_seq(board, depth, play_as_o=False, constants=None):
    """
    Search for the best move without using multiprocessing.
    :param board:
    :param depth:
    :param play_as_o:
    :param constants:
    :return:
    """

    if len(board.children) == 0:
        board.add_children()

    moves_and_evals = []
    for i in board.children:
        moves_and_evals.append(
            [i, minimax(i, depth - 1, float("-inf"), float("inf"), play_as_o)]
        )

    print(list(moves_and_evals))

    if not play_as_o:

        return max(moves_and_evals, key=lambda x: (x[1], x[0].inf_depth))
    else:
        return min(moves_and_evals, key=lambda x: (x[1], -x[0].inf_depth))


def minimax_search(board, depth, play_as_o=False, constants=None):
    """
    Search for the best move for X in the given board.
    :param play_as_o:
    :param board: Node, the board to search
    :param depth: int, the depth of moves to search
    :return: Node, the best move
    """

    if len(board.children) == 0:
        board.add_children()

    # Pruning might not be working optimally b/c the first set of child nodes are all kept separate
    minimax_partial = partial(
        minimax,
        depth=depth - 1,
        alpha=float("-inf"),
        beta=float("inf"),
        maximizing_player=play_as_o,
        constants=constants,
    )

    with Pool() as p:
        evals = p.map(minimax_partial, board.children)

    moves_and_evals = list(zip(board.children, evals))

    print(moves_and_evals)
    if not play_as_o:

        return max(moves_and_evals, key=lambda x: (x[1], x[0].inf_depth))
    else:
        return min(moves_and_evals, key=lambda x: (x[1], -x[0].inf_depth))


def minimax_search_move(board, depth, play_as_o=False, constants=None):
    """
    Searches for a move using minimax when given a node.
    :param board: Node, the board to search from
    :param depth: int, the depth to search to
    :param play_as_o: bool, default=False, is the program playing as O
    :return: [board to move on, piece to play]
    """
    return minimax_search(board, depth, play_as_o, constants)[0].board.previous_move


def minimax_search_move_from_board(board, depth, play_as_o=False, constants=None):
    """
    Searches for a move using minimax when given a GameState.
    :param board: GameState, the board to search from
    :param depth: int, the depth to search to
    :param play_as_o: bool, default=False, is the program playing as O
    :return: [board to move on, piece to play]
    """

    return minimax_search_move(Node(board), depth, play_as_o, constants)


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

        # Save eval once calculated
        self._eval = None

        # Save if this node has been previously pruned
        self.pruned = False

        # Used to delay forced losses and take forced wins
        self.inf_depth = 0

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
        :param final_score: bool, If true, children with n=0 with return -inf; If false, children with n=0 will return inf
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
        result = temp_board.game_result()

        # If win
        if (result == 1 and play_x) or (result == -1 and not play_x):
            return 1

        # If lose
        elif (result == -1 and play_x) or (result == 1 and not play_x):
            return -1

        # If tie
        elif result == 0:
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
        result = self.board.game_result()

        # If the result is not none, the node is terminal
        if result != 2:
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

    # def calc_eval_from_children(self, get_min=False):
    #     """
    #     Calculates the evaluation from its children's evaluations, recursively.
    #     If there are no children, return its heuristic evaluation.
    #     :param get_min: Set to true to get the lowest child evaluation
    #     :return: float; the evaluation based on children
    #     """
    #
    #     # If no children, return heuristic evaluation
    #     if len(self.children) == 0:
    #         return self.eval
    #
    #     # Get all the evaluations
    #     values = []
    #     for i in self.children:
    #         values.append(i.calc_eval_from_children(not get_min))
    #
    #     # Flip get_min with each turn
    #     if get_min:
    #         return min(values)
    #
    #     return max(values)

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

        if self._eval is not None:
            return self._eval

        # If the game is over, return appropriate value
        result = self.board.game_result()
        if result == 1:
            result = float("inf")
        elif result == -1:
            result = float("-inf")
        # If tie return 0
        elif result == 0:
            result = 0

        else:

            result = eval_board(self.board.board)

        self._eval = result
        return result

    def eval_constants(self, constants):

        # If the game is over, return appropriate value
        result = self.board.game_result()
        if result == 1:
            result = float("inf")
        elif result == -1:
            result = float("-inf")
        # If tie return 0
        elif result == 0:
            result = 0

        else:

            result = eval_board(self.board.board, constants)

        self._eval = result
        return result

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

    if verbose:
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


def mcts_search_move(start_board, depth, play_as_o=False, constants=None):
    """

    :param start_board: Board as a GameState
    :param depth: Number of iterations to perform
    :param play_as_o: bool
    :param constants: Not used
    :return:
    """

    board = start_board.copy_board()

    if play_as_o:
        board.board = flip_board(board.board)
        board.to_move = 1

    board_node = Node(board)

    return mcts(board_node, depth).board.previous_move
