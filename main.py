import mcts
import cProfile
import pickle


class GameState:
    def __init__(self):

        # List of the board
        self.board = []

        # Which piece's turn is it?
        self.to_move = "X"

        # Which board must they move on, none if there is no requirement
        self.board_to_move = None
        for i in range(9):
            # Add an empty board
            self.board.append([None] * 9)

        self.previous_move = None

    def copy_board(self):
        """
        Returns an exact new_copy of the current board, much faster than deepcopy
        :return: GameState
        """

        return pickle.loads(pickle.dumps(self, -1))

    @staticmethod
    def check_win(board):
        """
        Checks if the given board is a win (ie three in a row)
        :param board: A list of the board
        :return: 'X' if x is winning, 'O' if o is winning, None if neither is winning and the game goes on, False if a tie
        """

        # If the board is empty, the game goes on
        if set(board) == [None, None, None, None, None, None, None, None, None]:
            return None

        def check_rows_and_columns_and_diagonals(b):
            # Check each row
            for i in range(3):
                row = b[i * 3 : i * 3 + 3]

                # Check for win
                if row == ["X", "X", "X"]:
                    return "X"

                # Check for loss
                if row == ["O", "O", "O"]:
                    return "O"

            # Check each column
            for i in range(3):
                row = [b[i], b[i + 3], b[i + 6]]

                # Check for win
                if row == ["X", "X", "X"]:
                    return "X"

                # Check for loss
                if row == ["O", "O", "O"]:
                    return "O"

            # Check both diagonals
            for row in [
                [b[0], b[4], b[8]],
                [b[2], b[4], b[6]],
            ]:

                # Check for win
                if row == ["X", "X", "X"]:
                    return "X"

                # Check for loss
                if row == ["O", "O", "O"]:
                    return "O"

        result = check_rows_and_columns_and_diagonals(board)

        if result is not None:
            return result

        # If the board is filled, tie; if not, the game continues
        if None in board:
            return None

        return False

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

    @property
    def game_result(self):
        """
        Gets the status of the game.

        Possible statuses are:
        'X' - the game is over and X wins
        'O' - the game is over and O wins
        None - the game is in progress
        False - the game is a tie
        :return: The result of the game; False if tie; and None if the game is in progress
        """

        full_board_results = []

        for i in self.board:
            full_board_results.append(self.check_win(i))

        return self.check_win(full_board_results)

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
        if self.to_move == "X":
            self.to_move = "O"
        elif self.to_move == "O":
            self.to_move = "X"
        else:
            raise RuntimeError(f"Unexpected value of self.to_move '{self.to_move}'")

        # Update board

        # If the board in the position of the move is finished, there is no required next board
        if self.check_win(self.board[spot]) is None:
            self.board_to_move = spot
        else:
            self.board_to_move = None

        # Update previous move
        self.previous_move = [board, spot]

    def all_possible_moves(self):
        """
        Returns a list of all possible boards that can be made from moves played on this board
        :return: List[GameState]
        """

        possibilities = []

        # If they can move anywhere
        if self.board_to_move is None:
            board_index = -1

            # Loop every board
            for b in self.board:

                board_index += 1
                spot_index = -1

                # If the board is won, no moves can be played on it
                if self.check_win(b) is not None:
                    continue

                # Loop every spot
                for spot in b:
                    spot_index += 1

                    # Add a move for every empty spot
                    if spot is None:
                        new_move = self.copy_board()

                        new_move.move(board_index, spot_index)

                        possibilities.append(new_move)

        # If they must move on a specific board
        else:

            spot_index = -1
            for spot in self.board[self.board_to_move]:
                spot_index += 1

                if spot is None:
                    new_move = self.copy_board()
                    new_move.move(self.board_to_move, spot_index)
                    possibilities.append(new_move)

        return possibilities

    @staticmethod
    def board_to_str(b):
        result = []
        n = 3

        # Replace empty squares with spaces for display
        index = -1
        for i in b:
            index += 1
            if i is None:
                b[index] = " "

        # Loop through each tow
        for row in [b[i : i + n] for i in range(0, len(b), n)]:
            result.append("|".join(row))
            result[-1] = result[-1].replace("None", " ")
            result.append("-" * len(result[0 - 1]))

        # Chop off the last set of '--------'
        return "\n".join(result[:-1])

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

                    # Replace None with empty space
                    piece_char = self.board[board_index][piece_index] or " "

                    result += piece_char
                    result += " | "
                result = result[:-3] + "\\\\ "

            if (row + 1) % 3 != 0:
                result += f"\n{'---------   ' * 3}\n"
            else:
                result += "\n=================================\n"

        return result.replace("O", "\033[94mO\033[0m").replace("X", "\033[31mX\033[0m")

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

    while b.game_result is None:

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

        if b.game_result is not None:

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
        print(f"RESULT {b.game_result}")

    return b.game_result


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

    b.to_move = "X"

    # b.move(4, 4)
    # b.move(4, 0)

    # b.board_to_move = 5
    # b.board = [
    #     [None, None, None, None, None, None, None, None, None],
    #     [None, None, None, None, None, None, None, None, None],
    #     ["X", "X", "X", None, None, None, None, None, None],
    #     [None, None, None, None, None, None, None, None, None],
    #     ["X", "X", "X", None, None, None, None, None, None],
    #     [None, None, "X", None, "X", None, None, None, None],
    #     ["X", "X", None, "X", None, None, None, "O", "O"],
    #     ["O", "O", "O", None, None, None, None, None, None],
    #     ["O", "O", "O", None, None, None, None, None, None],
    # ]

    # b.board_to_move = 3
    # b.board = [
    #     ["X", "X", None, "O", None, "X", None, None, None],
    #     ["O", "O", None, "O", None, "X", None, None, None],
    #     ["X", "X", "X", None, None, None, None, None, None],
    #     ["O", "X", None, "X", None, None, None, None, None],
    #     ["X", "X", "X", None, None, None, None, None, None],
    #     ["O", "X", "X", "O", "X", None, None, None, None],
    #     ["X", "X", None, "X", None, None, None, "O", "O"],
    #     ["O", "O", "O", None, None, None, None, None, None],
    #     ["O", "O", "O", None, None, None, None, None, None],
    # ]
    #
    # b.to_move = "X"
    # b.board_to_move = 5
    # b.board = mcts.flip_board(
    #     [
    #         ["X", "X", None, "O", None, "X", None, None, None],
    #         ["O", "O", None, "O", None, "X", None, None, None],
    #         ["X", "X", "X", None, None, None, None, None, None],
    #         ["O", "X", None, "X", None, "X", None, None, None],
    #         ["X", "X", "X", None, None, None, None, None, None],
    #         ["O", "X", "X", "O", "X", None, None, None, None],
    #         ["X", "X", None, "X", None, None, None, "O", "O"],
    #         ["O", "O", "O", None, None, None, None, None, None],
    #         ["O", "O", "O", None, None, None, None, None, None],
    #     ]
    # )

    # TEST BOARD
    # b.board_to_move = 4
    # b.board = [
    #     ["O", "X", None, None, "X", None, None, "X", None],
    #     [None, None, "O", None, None, "O", None, "X", None],
    #     [None, "O", "O", "X", "X", "O", None, None, "X"],
    #     [None, None, "X", "O", None, "X", None, None, "X"],
    #     ["O", None, "O", None, "X", None, None, None, "O"],
    #     [None, None, "X", "O", "O", "X", None, None, "O"],
    #     ["X", None, None, None, None, None, None, None, None],
    #     [None, None, None, None, None, None, "O", "O", "O"],
    #     [None, "X", "X", "O", None, "X", None, "X", "O"],
    # ]
    #
    # print(b)

    # Ongoing game
    b.board_to_move = 6
    b.board = [
        ["O", None, None, "X", "X", "X", None, None, "O", None],
        ["X", None, "O", None, None, None, None, "X", None],
        ["O", "X", None, "X", "X", None, "O", None, None],
        ["O", "O", "O", None, "X", None, None, "X", None],
        [None, None, "X", "O", "X", "O", "O", "O", "O"],
        [None, None, None, None, "O", "X", "O", None, "X"],
        [None, None, None, None, "X", None, None, None, "X"],
        [None, "O", "O", "O", None, "X", None, "X", None],
        ["X", None, None, "X", None, "O", None, None, "O"],
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
    # print(b.check_win(b.board[5]))
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

    while b.game_result is None:

        # print(
        #     f"Thinking... {current_game_node.descendants} nodes are in the tree and {current_game_node.n} iterations saved"
        # )

        # Have the computer play a move
        # current_game_node = mcts.mcts(current_game_node, 1000, 20)
        # if current_game_node.board.board_to_move is None:
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

        if b.game_result is not None:
            print("You have lost!")
            break

        print("Your move:")

        if len(current_game_node.children) == 0:
            current_game_node.add_children()
            print("Adding children")

        asking_for_move = True
        while asking_for_move:
            if b.board_to_move is not None:
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

    if b.game_result == "O":
        print("You have won!!")
