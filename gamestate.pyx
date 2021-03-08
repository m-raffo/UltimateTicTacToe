import mcts
import cProfile
import pickle
import numpy as np
cimport numpy as np
import cython

cdef class GameState:
    cdef int to_move, board_to_move

    cdef int [:,:] board
    cdef int [:] previous_move

    def __init__(self):

        # List of the board
        self.board = np.full((9, 9), 0, dtype=np.int32)

        # Which piece's turn is it?
        # 1 = X
        # -1 = O
        self.to_move = 1

        # Which board must they move on, none if there is no requirement
        self.board_to_move = -1

        self.previous_move = np.array([-1, -1], dtype=np.int32)

    def __getstate__(self):
        return [np.asarray(self.board), self.to_move, self.board_to_move, np.asarray(self.previous_move)]

    def __setstate__(self, x):
        # try:
        #     self.board.base, self.to_move, self.board_to_move, self.previous_move.base = x
        # except AttributeError:
        #     _, self.to_move, self.board_to_move, _ = x
        #     self.board = np.full((9, 9), 0, dtype=np.int32)
        #     self.previous_move = np.array([-1, -1], dtype=np.int32)
        self.board, self.to_move, self.board_to_move, self.previous_move = x


    def copy_board(self):
        """
        Returns an exact new_copy of the current board, much faster than deepcopy
        :return: GameState
        """

        cdef GameState new_board = GameState()

        new_board.board[:] = self.board
        new_board.to_move = self.to_move
        new_board.board_to_move = self.board_to_move
        new_board.previous_move[:] = self.previous_move

        return new_board


    @property
    def board_to_move(self):
        return self.board_to_move

    @board_to_move.setter
    def board_to_move(self, value):
        self.board_to_move = value

    @property
    def previous_move(self):
        return self.previous_move

    @previous_move.setter
    def previous_move(self, value):
        self.previous_move = value

    @property
    def board(self):
        return self.board

    @board.setter
    def board(self, value):
        self.board = value


    @property
    def to_move(self):
        return self.to_move

    @to_move.setter
    def to_move(self, value):
        self.to_move = value


    def check_valid_move(self, board, spot):
        """
        Checks if the move is valid based on current conditions

        WARNING: This method is not fully implemented yet!

        :param board: int between 0 and 8 to move on
        :param spot: int between 0 and 8 for the spot to play on
        :return: True if valid, false if not
        """

        if 0 <= board <= 8 and 0 <= spot <= 8:
            return True

        return False

    cpdef game_result(self):
        """
        Gets the status of the game.

        Possible statuses are:
        1 - the game is over and X wins
        -1 - the game is over and O wins
        2 - the game is in progress
        0 - the game is a tie
        :return: The result of the game; False if tie; and None if the game is in progress
        """

        cdef int [:] full_board_results = np.full(9, 0, dtype=np.int32)
        cdef int index = -1


        for i in self.board:
            index += 1
            miniboard_result = mcts.check_win(i)

            # If the game is ongoing, the space is effectively empty
            if miniboard_result != 2:
                full_board_results[index] = miniboard_result

        return mcts.check_win(full_board_results)

    def move(self, board, spot):
        """
        Make the given move on the board.
        :param board: int between 0 and 8 to move on
        :param spot: int between 0 and 8 to put the piece on the given board
        :return: None
        """

        # Check that the move is valid
        if not self.check_valid_move(board, spot):
            return False

        self.board[board][spot] = self.to_move

        # Update to move
        self.to_move *= -1

        # Update board

        # If the board in the position of the move is finished, there is no required next board
        if mcts.check_win(self.board[spot]) == 2:
            self.board_to_move = spot
        else:
            self.board_to_move = -1

        # Update previous move
        self.previous_move = np.array([board, spot], dtype=np.int32)

    def all_possible_moves(self):
        """
        Returns a list of all possible boards that can be made from moves played on this board
        :return: List[GameState]
        """

        possibilities = []

        # If they can move anywhere
        if self.board_to_move == -1:
            board_index = -1

            # Loop every board
            for b in self.board:

                board_index += 1
                spot_index = -1

                # If the board is won, no moves can be played on it
                if mcts.check_win(b) != 2:
                    continue

                # Loop every spot
                for spot in b:
                    spot_index += 1

                    # Add a move for every empty spot
                    if spot == 0:
                        new_move = self.copy_board()

                        new_move.move(board_index, spot_index)

                        possibilities.append(new_move)

        # If they must move on a specific board
        else:

            spot_index = -1
            for spot in self.board[self.board_to_move]:
                spot_index += 1

                if spot == 0:
                    new_move = self.copy_board()
                    new_move.move(self.board_to_move, spot_index)
                    possibilities.append(new_move)

        return possibilities

    # @staticmethod
    # def board_to_str(b):
    #     result = []
    #     n = 3
    #
    #     # Replace empty squares with spaces for display
    #     index = -1
    #     for i in b:
    #         index += 1
    #         if i == 0:
    #             b[index] = " "
    #
    #     # Loop through each tow
    #     for row in [b[i : i + n] for i in range(0, len(b), n)]:
    #         result.append("|".join(row))
    #         result[-1] = result[-1].replace("None", " ")
    #         result.append("-" * len(result[0 - 1]))
    #
    #     # Chop off the last set of '--------' and replace numbers with letters
    #     return (
    #         "\n".join(result[:-1]).replace("-1", -1).replace("1", "X").replace("0", " ")
    #     )

    @staticmethod
    def absolute_index_to_board_and_piece(index):
        """
        Gets an absolute piece index to a board and piece
        :param index: The index to be found
        :return: (board_index, piece_index)
        """

        # i = index
        # gx = global x value (independent of board)
        # gy = same
        #
        # lx = local x value (within board)
        # ly = same
        #
        # bx = x value of the whole board
        # by = same
        #
        # pi = piece index
        # bi = board index

        i = index

        gx = i % 9
        gy = int(i / 9)

        lx = gx % 3
        ly = gy % 3

        bx = int((i % 9) / 3)
        by = int(i / 27)

        pi = ly * 3 + lx
        bi = by * 3 + bx

        return bi, pi

    def __str__(self):
        result = f"{self.to_move} to move\nRequired board: {self.board_to_move}\n"

        for row in range(9):

            for board_row in range(3):
                for col in range(3):
                    absolute_piece_index = (row * 9) + (board_row * 3) + col

                    board_index, piece_index = self.absolute_index_to_board_and_piece(
                        absolute_piece_index
                    )

                    # Replace 0 with empty space
                    piece_char = self.board[board_index][piece_index] or " "

                    result += str(piece_char)
                    result += " | "
                result = result[:-3] + "\\\\ "

            if (row + 1) % 3 != 0:
                result += f"\n{'---------   ' * 3}\n"
            else:
                result += "\n=================================\n"

        return result.replace("-1", "\033[94mO\033[0m").replace("1", "\033[31mX\033[0m")

    # Save the previous move in the format [board, spot]
    # Only for display purposes


def computer_vs_computer(
    move_function1,
    depth1,
    constants1,
    move_function2,
    depth2,
    constants2,
    starting_gamestate=None,
    print_moves=False,
):

    if starting_gamestate is not None:
        b = starting_gamestate
    else:
        b = GameState()

    while b.game_result() == 2:

        # Calculate the first move
        comp1_move = move_function1(b.copy_board(), depth1, constants=constants1)
        b.move(*comp1_move)

        if print_moves:
            print("COMPUTER 1 MOVE:")
            print(b.previous_move)
            print(mcts.detail_eval(b.board))
            # print(f"MINIMAX: {search_results[1]}")
            print(b.board)
            print(b)

        if b.game_result() != 2:

            break

        # Calculate the second move
        comp2_move = move_function2(b.copy_board(), depth2, True, constants=constants2)
        b.move(*comp2_move)

        if print_moves:
            print("COMPUTER 2 MOVE:")
            print(b.previous_move)
            print(mcts.detail_eval(b.board))
            # print(f"MINIMAX: {search_results[1]}")
            print(b.board)
            print(b)

    if print_moves:
        print("The game is over!")
        print(f"RESULT {b.game_result()}")

    return b.game_result()


if __name__ == "__main__":

    c1 = {"c1": 2, "c2": 1, "cw": 10, "cl": 0}

    c2 = {"c1": 0.05, "c2": 3, "cw": 2, "cl": 0}

    # computer_vs_computer(
    #     mcts.minimax_search_move_from_board,
    #     2,
    #     c1,
    #     mcts.mcts_search_move,
    #     10000,
    #     c2,
    #     print_moves=True,
    # )

    # computer_vs_computer(
    #     mcts.mcts_search_move,
    #     30000,
    #     c2,
    #     mcts.minimax_search_move_from_board,
    #     2,
    #     c1,
    #     print_moves=True,
    # )
    #
    # exit()
    # computer_vs_computer()
    #
    # exit()

    b = GameState()

    b.to_move = 1

    # b.move(4, 4)
    # b.move(4, 0)

    # b.board_to_move = 5
    # b.board = [
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [1, 1, 1, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [1, 1, 1, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 1, 0, 1, 0, 0, 0, 0],
    #     [1, 1, 0, 1, 0, 0, 0, -1, -1],
    #     [-1, -1, -1, 0, 0, 0, 0, 0, 0],
    #     [-1, -1, -1, 0, 0, 0, 0, 0, 0],
    # ]

    # b.board_to_move = 3
    # b.board = [
    #     [1, 1, 0, -1, 0, 1, 0, 0, 0],
    #     [-1, -1, 0, -1, 0, 1, 0, 0, 0],
    #     [1, 1, 1, 0, 0, 0, 0, 0, 0],
    #     [-1, 1, 0, 1, 0, 0, 0, 0, 0],
    #     [1, 1, 1, 0, 0, 0, 0, 0, 0],
    #     [-1, 1, 1, -1, 1, 0, 0, 0, 0],
    #     [1, 1, 0, 1, 0, 0, 0, -1, -1],
    #     [-1, -1, -1, 0, 0, 0, 0, 0, 0],
    #     [-1, -1, -1, 0, 0, 0, 0, 0, 0],
    # ]
    #
    # b.to_move = 1
    # b.board_to_move = 5
    # b.board = mcts.flip_board(
    #     [
    #         [1, 1, 0, -1, 0, 1, 0, 0, 0],
    #         [-1, -1, 0, -1, 0, 1, 0, 0, 0],
    #         [1, 1, 1, 0, 0, 0, 0, 0, 0],
    #         [-1, 1, 0, 1, 0, 1, 0, 0, 0],
    #         [1, 1, 1, 0, 0, 0, 0, 0, 0],
    #         [-1, 1, 1, -1, 1, 0, 0, 0, 0],
    #         [1, 1, 0, 1, 0, 0, 0, -1, -1],
    #         [-1, -1, -1, 0, 0, 0, 0, 0, 0],
    #         [-1, -1, -1, 0, 0, 0, 0, 0, 0],
    #     ]
    # )

    # TEST BOARD
    # b.board_to_move = 4
    # b.board = [
    #     [-1, 1, 0, 0, 1, 0, 0, 1, 0],
    #     [0, 0, -1, 0, 0, -1, 0, 1, 0],
    #     [0, -1, -1, 1, 1, -1, 0, 0, 1],
    #     [0, 0, 1, -1, 0, 1, 0, 0, 1],
    #     [-1, 0, -1, 0, 1, 0, 0, 0, -1],
    #     [0, 0, 1, -1, -1, 1, 0, 0, -1],
    #     [1, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, -1, -1, -1],
    #     [0, 1, 1, -1, 0, 1, 0, 1, -1],
    # ]
    #
    # print(b)

    # Ongoing game
    b.board_to_move = 6
    b.board = [
        [-1, 0, 0, 1, 1, 1, 0, 0, -1, 0],
        [1, 0, -1, 0, 0, 0, 0, 1, 0],
        [-1, 1, 0, 1, 1, 0, -1, 0, 0],
        [-1, -1, -1, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, -1, 1, -1, -1, -1, -1],
        [0, 0, 0, 0, -1, 1, -1, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, -1, -1, -1, 0, 1, 0, 1, 0],
        [1, 0, 0, 1, 0, -1, 0, 0, -1],
    ]

    # b.move(4, 5)
    # b.move(5, 1)
    # b.move(1, 5)
    # b.move(5, 4)
    # b.move(4, 4)
    # b.move(3, 7)
    # b.move(4, 3)

    # print(b)
    #
    current_game_node = mcts.Node(b)

    current_game_node.add_children()
    #
    # print(mcts.eval_board(b.board))
    #
    # print(mcts.minimax(b, 3))

    #
    # cProfile.run("mcts.mcts(current_game_node, 10000)")
    #
    # exit()
    # print(mcts.check_win(b.board[5]))
    # exit()

    # a = mcts.Node(b)
    # a.add_children()
    #
    # for i in a.children:
    #     i.add_children()
    #
    # print(a.descendants)

    # cProfile.run("print(mcts.minimax_search(current_game_node, 7, False))")
    # exit()

    # computer_vs_computer()

    while b.game_result() == 2:

        # print(
        #     f"Thinking... {current_game_node.descendants} nodes are in the tree and {current_game_node.n} iterations saved"
        # )

        # Have the computer play a move
        # current_game_node = mcts.mcts(current_game_node, 1000, 20)
        # if current_game_node.board.board_to_move == -1:
        #     current_game_node = mcts.minimax_search(current_game_node, 3)
        # else:
        #     current_game_node = mcts.minimax(current_game_node, 4)

        search_results = mcts.minimax_search(current_game_node, 8, False)
        current_game_node = search_results[0]

        # current_game_node = mcts.mcts(current_game_node, 30000)

        b = current_game_node.board
        print("COMPUTER MOVE:")
        print(b.previous_move)
        print(mcts.detail_eval(b.board))
        print(f"MINIMAX: {search_results[1]}")
        print(b.board)
        print(b)

        if b.game_result() != 2:
            print("You have lost!")
            break

        print("Your move:")

        if len(current_game_node.children) == 0:
            current_game_node.add_children()
            print("Adding children")

        asking_for_move = True
        while asking_for_move:
            if b.board_to_move != -1:
                print(f"Board >> {b.board_to_move}")
                user_board = b.board_to_move
            else:
                user_board = int(input("Board >> "))
            user_spot = int(input("Spot >> "))
            test_board = b.copy_board()

            test_board.move(user_board, user_spot)

            # In order to avoid restarting the node tree every time, find the node among current_game_node.children with
            # the board that matches the player's move
            found_board = False
            for i in current_game_node.children:

                # If the board matches, use this as the new board
                if i.board.board == test_board.board:
                    current_game_node = i
                    found_board = True

            if not found_board:
                print(
                    "Your move was not found as a valid move. Are you sure you entered it correctly?"
                )
                print(current_game_node.children)
            else:
                asking_for_move = False
                b.move(user_board, user_spot)

        print(mcts.detail_eval(b.board))
        print(b.board)
        print(b)

    if b.game_result() == -1:
        print("You have won!!")
