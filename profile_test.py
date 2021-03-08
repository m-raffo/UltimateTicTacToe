import cProfile
from gamestate import GameState
import mcts
import timeit

import mctsold
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
    ]
)

print(b)

current_game_node = mcts.Node(b)

current_game_node.add_children()

print(current_game_node)
for i in current_game_node.children:
    print(i)


if __name__ == "__main__":
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
    cy_command = "print(mcts.minimax_search_seq(current_game_node, 6, False))"

    # cProfile.run(cy_command)
    cProfile.run(cy_command)
    # cy = timeit.timeit(cy_command)
    # py = timeit.timeit(py_command)
    #
    # print(cy, py)
    # print(f"Cython is {round(cy/py)}x faster")
