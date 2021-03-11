import cProfile
from gamestate import GameState
import mcts
import timeit

import numpy as np

# TEST BOARD

b = GameState()
b.board_to_move = 4
b.board = np.array(
    [
        [-1, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, -1, 0, 0, -1, 0, 1, 0],
        [0, -1, -1, 1, 1, -1, 0, 0, 1],
        [0, 0, 1, -1, 0, 1, 0, 0, 1],
        [-1, 0, -1, 0, 1, 0, 0, 0, -1],
        [0, 0, 1, -1, -1, 1, 0, 0, -1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -1, -1, -1],
        [0, 1, 1, -1, 0, 1, 0, 1, -1],
    ],
    dtype=np.int32,
)

print(b)

current_game_node = mcts.Node(b)

current_game_node.add_children()


if __name__ == "__main__":

    b = GameState()
    b.move(4, 5)

    print(
        timeit.timeit(
            # "a = mcts.eval_board(myboard.board)"
            "\nb = myboard.all_possible_moves()",
            number=1,
            setup="""
from gamestate import GameState
import mcts
myboard = GameState()
myboard.move(5, 0)
myboard.move(4, 5)
myboard.move(0, 1)
myboard.move(1, 1)
myboard.move(1, 8)
myboard.move(8, 4)
myboard.move(4, 3)
myboard.move(3, 0)
myboard.move(0, 0)
myboard.move(0, 6)
myboard.move(6, 3)
myboard.move(3, 1)
myboard.move(1, 2)
    """,
        )
    )

    exit()

    #     a = np.array([0, 0, 0])
    #     b = np.array([0, 0, 0])
    #
    #     c = [0, 0, 0]
    #     d = [0, 0, 0]
    #
    #     print("Starting timeit")
    #
    #     print(
    #         timeit.timeit(
    #             """a = numpy.array([0, 0, 0])
    # b = numpy.array([0, 0, 0])
    # numpy.array_equal(a, b)""",
    #             setup="""import numpy
    #
    # print('starting euqlaity check')""",
    #         )
    #     )
    #     print(
    #         timeit.timeit(
    #             "c == d",
    #             setup="""c = [0, 0, 0]
    # d = [0, 0 ,0]""",
    #         )
    #     )
    #
    #     exit()

    py_command = "print(mctsold.minimax_search_seq(current_game_node, 2, False))"
    cy_command = "print(mcts.minimax_search_seq_variable_pruning(current_game_node, [3, 4, 6], False))"

    cy_command2 = (
        "print(mcts.minimax_search_seq_pruning(current_game_node, 3, 6, False))"
    )

    # cProfile.run(cy_command)
    cProfile.run(cy_command2)
    # cy = timeit.timeit(cy_command)
    # py = timeit.timeit(py_command)
    #
    # print(cy, py)
    # print(f"Cython is {round(cy/py)}x faster")
