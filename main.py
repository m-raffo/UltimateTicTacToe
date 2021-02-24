import numpy as np
import copy


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

    def copy_board(self):
        """
        Returns an exact new_copy of the current board
        :return: GameState
        """

        new_copy = GameState()

        new_copy.board = copy.deepcopy(self.board)
        new_copy.to_move = self.to_move
        new_copy.board_to_move = self.board_to_move

        return new_copy

    @staticmethod
    def check_win(board):
        """
        Checks if the given board is a win (ie three in a row)
        :param board: A list of the board
        :return: 'X' if x is winning, 'O' if o is winning, None if neither is winning and the game goes on, False if a tie
        """

        def check_rows(b):
            for row in b:
                if len(set(row)) == 1:
                    return row[0]
            return None

        def check_diagonals(b):
            if len(set([b[i][i] for i in range(len(b))])) == 1:
                return b[0][0]
            if len(set([b[i][len(b) - i - 1] for i in range(len(b))])) == 1:
                return b[0][len(b) - 1]
            return None

        # Check both rows and columns, return if there is a winner
        board_width = 3
        square_board = [
            board[i : i + board_width] for i in range(0, len(board), board_width)
        ]

        for new_board in [square_board, np.transpose(square_board)]:
            result = check_rows(new_board)

            if result:
                return result

        # Check diagonals
        result = check_diagonals(new_board)
        if result:
            return result

        # If the board is filled, tie; if not, the game continues
        if None in board:
            return None

        return False

    def check_valid_move(self, board, spot):
        """
        Checks if the move is valid based on current conditions

        WARNING: This method is not implemented yet!

        :param board: int between 0 and 8 to move on
        :param spot: int between 0 and 8 for the spot to play on
        :return: True if valid, false if not
        """

        return True

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
            print(f"WHY IS TO MOVE SET TO BE '{self.to_move}'????")

        # Update board

        # If the board in the position of the move is finished, there is no required next board
        if self.check_win(self.board[spot]) is None:
            self.board_to_move = spot
        else:
            self.board_to_move = None

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
        result = ""

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

        return result


if __name__ == "__main__":
    b = GameState()

    b.move(0, 1)

    print(b)

    # input("Press enter to print all moves")

    next_board = b.all_possible_moves()[1]
    print("NEXT BOARD")
    print(next_board)
    print("+" * 20)
    for i in next_board.all_possible_moves():
        print(i)
