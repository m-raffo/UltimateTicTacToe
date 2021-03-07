import cProfile
from main import GameState
import mcts

# TEST BOARD

b = GameState()
b.board_to_move = 4
b.board = [
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


current_game_node = mcts.Node(b)

current_game_node.add_children()

if __name__ == "__main__":
    cProfile.run("print(mcts.minimax_search_seq(current_game_node, 5, False))")
