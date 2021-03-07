import cProfile
from main import GameState
import mcts

# TEST BOARD

b = GameState()
b.board_to_move = 4
b.board = [
    ["O", "X", None, None, "X", None, None, "X", None],
    [None, None, "O", None, None, "O", None, "X", None],
    [None, "O", "O", "X", "X", "O", None, None, "X"],
    [None, None, "X", "O", None, "X", None, None, "X"],
    ["O", None, "O", None, "X", None, None, None, "O"],
    [None, None, "X", "O", "O", "X", None, None, "O"],
    ["X", None, None, None, None, None, None, None, None],
    [None, None, None, None, None, None, "O", "O", "O"],
    [None, "X", "X", "O", None, "X", None, "X", "O"],
]


current_game_node = mcts.Node(b)

current_game_node.add_children()

if __name__ == "__main__":
    cProfile.run("print(mcts.minimax_search_seq(current_game_node, 7, False))")
