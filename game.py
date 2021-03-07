import pygame

import main
import mcts

from pygame.locals import (
    K_ESCAPE,
    KEYDOWN,
    QUIT,
    MOUSEBUTTONDOWN,
)


# Initialize pygame
pygame.init()

# Define constants for the screen width and height

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
BOARD_BUFFER_X = 25
BOARD_BUFFER_Y = 25


HUMAN_PLAY_AS_O = True
DEPTH = 2


def draw_game(screen, game):
    draw_board(screen)

    board_index = -1
    for miniboard in game.board:
        board_index += 1

        miniboard_status = game.check_win(miniboard)

        board_x = board_index % 3
        board_y = int(board_index / 3)

        if miniboard_status is not None:

            board_start_x = (
                SCREEN_WIDTH - 2 * BOARD_BUFFER_X
            ) / 3 * board_x + BOARD_BUFFER_X

            board_start_y = (
                SCREEN_HEIGHT - 2 * BOARD_BUFFER_Y
            ) / 3 * board_y + BOARD_BUFFER_Y

            if miniboard_status == 1:
                img = big_font.render("X", True, (200, 100, 100))

            elif miniboard_status == -1:
                img = big_font.render("O", True, (100, 100, 200))

            else:
                img = big_font.render("T", True, (100, 100, 100))

            board_width = (SCREEN_WIDTH - 2 * BOARD_BUFFER_X) / 3

            board_height = (SCREEN_HEIGHT - 2 * BOARD_BUFFER_Y) / 3

            imgx = board_start_x + 0.5 * board_width - 0.5 * img.get_width()
            imgy = board_start_y + 0.5 * board_height - 0.5 * img.get_height()

            screen.blit(img, (imgx, imgy))
        else:
            spot_index = -1
            for spot in miniboard:
                spot_index += 1

                if spot == 1:
                    draw_move(screen, board_index, spot_index, "X", font)

                elif spot == -1:
                    draw_move(screen, board_index, spot_index, "O", font)

    if game.board_to_move is not None:
        if game.to_move == -1:
            box_board(screen, game.board_to_move, (100, 100, 200))
        else:
            box_board(screen, game.board_to_move, (200, 100, 100))

    pygame.display.flip()


def box_board(screen, board_index, to_move_color):
    board_x = board_index % 3
    board_y = int(board_index / 3)

    board_start_x = (SCREEN_WIDTH - 2 * BOARD_BUFFER_X) / 3 * board_x + BOARD_BUFFER_X

    board_start_y = (SCREEN_HEIGHT - 2 * BOARD_BUFFER_Y) / 3 * board_y + BOARD_BUFFER_Y

    board_width = (SCREEN_WIDTH - 2 * BOARD_BUFFER_X) / 3

    board_height = (SCREEN_HEIGHT - 2 * BOARD_BUFFER_Y) / 3

    pygame.draw.rect(
        screen,
        to_move_color,
        pygame.Rect(board_start_x, board_start_y, board_width, board_height),
        width=5,
    )


def draw_board(screen):
    for i in range(1, 3):
        line_x = (SCREEN_WIDTH - 2 * BOARD_BUFFER_X) / 3 * i + BOARD_BUFFER_X

        pygame.draw.line(
            screen,
            (0, 0, 0),
            (line_x, BOARD_BUFFER_Y),
            (line_x, SCREEN_HEIGHT - BOARD_BUFFER_Y),
            width=5,
        )

    for i in range(1, 3):
        line_y = (SCREEN_HEIGHT - 2 * BOARD_BUFFER_Y) / 3 * i + BOARD_BUFFER_Y

        pygame.draw.line(
            screen,
            (0, 0, 0),
            (BOARD_BUFFER_X, line_y),
            (SCREEN_WIDTH - BOARD_BUFFER_X, line_y),
            width=5,
        )

    line_width = (SCREEN_WIDTH - 2 * BOARD_BUFFER_X / 3) / 3

    line_height = (SCREEN_HEIGHT - 2 * BOARD_BUFFER_Y / 3) / 3

    # Draw the small boards
    for board_pos_x in range(3):

        for board_pos_y in range(3):

            board_start_x = (
                SCREEN_WIDTH - 2 * BOARD_BUFFER_X
            ) / 3 * board_pos_x + BOARD_BUFFER_X

            board_start_y = (
                SCREEN_HEIGHT - 2 * BOARD_BUFFER_Y
            ) / 3 * board_pos_y + BOARD_BUFFER_Y

            for line_number in range(1, 3):
                line_x = (
                    board_start_x
                    + (SCREEN_WIDTH - 2 * BOARD_BUFFER_X) / 9 * line_number
                )

                line_y = (
                    board_start_y
                    + (SCREEN_HEIGHT - 2 * BOARD_BUFFER_Y) / 9 * line_number
                )

                pygame.draw.line(
                    screen,
                    (0, 0, 0),
                    (line_x, board_start_y + BOARD_BUFFER_Y),
                    (line_x, board_start_y + line_height - BOARD_BUFFER_Y),
                )

                pygame.draw.line(
                    screen,
                    (0, 0, 0),
                    (board_start_x + BOARD_BUFFER_X, line_y),
                    (board_start_x + line_width - BOARD_BUFFER_X, line_y),
                )


def mouse_pos_to_board_and_piece(mouse_pos):
    x_coord = int(
        ((mouse_pos[0] - BOARD_BUFFER_X) / ((SCREEN_WIDTH - 2 * BOARD_BUFFER_X) / 9))
    )
    y_coord = int(
        ((mouse_pos[1] - BOARD_BUFFER_Y) / ((SCREEN_HEIGHT - 2 * BOARD_BUFFER_Y) / 9))
    )

    print(x_coord)
    print(y_coord)

    return absolute_coords_to_board_and_piece(x_coord, y_coord)


def absolute_coords_to_board_and_piece(xpos, ypos):

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

    i = xpos + 9 * ypos

    gx = i % 9
    gy = int(i / 9)

    lx = gx % 3
    ly = gy % 3

    bx = int((i % 9) / 3)
    by = int(i / 27)

    pi = ly * 3 + lx
    bi = by * 3 + bx

    return bi, pi


def draw_move(screen, board, piece, text, font, color=None):
    board_x = board % 3
    board_y = int(board / 3)

    piece_x = piece % 3
    piece_y = int(piece / 3)

    board_start_x = (SCREEN_WIDTH - 2 * BOARD_BUFFER_X) / 3 * board_x + BOARD_BUFFER_X

    board_start_y = (SCREEN_HEIGHT - 2 * BOARD_BUFFER_Y) / 3 * board_y + BOARD_BUFFER_Y

    board_width = (SCREEN_WIDTH - 2 * BOARD_BUFFER_X) / 3

    board_height = (SCREEN_HEIGHT - 2 * BOARD_BUFFER_Y) / 3

    piece_start_x = (
        board_start_x
        + BOARD_BUFFER_X
        + (board_width - 2 * BOARD_BUFFER_X) / 3 * piece_x
    )
    piece_start_y = (
        board_start_y
        + BOARD_BUFFER_Y
        + (board_height - 2 * BOARD_BUFFER_Y) / 3 * piece_y
    )

    if color is not None:
        img = font.render(text, True, color)

    else:
        if text == "O":

            img = font.render(text, True, (100, 100, 200))
        elif text == "X":
            img = font.render(text, True, (200, 100, 100))

        else:
            img = font.render(text, True, (0, 0, 0))

    draw_x = piece_start_x + 0.5 * (board_width / 3) - 0.5 * img.get_width()
    draw_y = piece_start_y + 0.5 * (board_height / 3) - 0.5 * img.get_height()

    screen.blit(img, (draw_x, draw_y))


def display_message(screen, text, color=(255, 255, 255), bg_color=(100, 100, 100)):
    img = med_font.render(text, True, color)

    rectx = SCREEN_WIDTH / 2 - img.get_width() / 2 - 10
    recty = SCREEN_HEIGHT / 2 - img.get_height() / 2 - 10

    width = img.get_width() + 20
    height = img.get_height() + 20

    # Draw the background
    pygame.draw.rect(
        screen,
        bg_color,
        pygame.Rect(rectx, recty, width, height),
    )

    # Draw the text
    screen.blit(img, (rectx + 10, recty + 10))
    pygame.display.flip()


if __name__ == "__main__":
    pygame.init()

    # Create the screen object

    # The size is determined by the constant SCREEN_WIDTH and SCREEN_HEIGHT

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    running = True

    screen.fill((255, 255, 255))

    draw_board(screen)

    font = pygame.font.SysFont(None, 65)

    big_font = pygame.font.SysFont(None, 425)
    med_font = pygame.font.SysFont(None, 200)

    small_font = pygame.font.SysFont(None, 30)

    is_players_move = False

    game = main.GameState()

    if HUMAN_PLAY_AS_O:
        game.move(4, 4)

    draw_game(screen, game)
    is_players_move = True
    game_running = True

    minimax_node = mcts.Node(game)

    minimax_results = None

    while running:

        if not is_players_move and game.game_result is None:
            if minimax_results.ready():
                p.close()
                p.terminate()

                is_players_move = True
                moves_and_evals = zip(minimax_node.children, minimax_results.get())

                if HUMAN_PLAY_AS_O:

                    minimax_node, current_eval = max(
                        moves_and_evals, key=lambda x: x[1]
                    )
                else:
                    minimax_node, current_eval = min(
                        moves_and_evals, key=lambda x: x[1]
                    )

                move = minimax_node.board.previous_move
                game.move(*move)
                # draw_move(screen, move[0], move[1], "X", font)

                screen.fill((255, 255, 255))

                draw_game(screen, game)

                print(game)

                result = game.game_result
                if result is not None:
                    if result == 1:
                        display_message(screen, "X WINS!")
                    elif result == -1:
                        display_message(screen, "O WINS!")
                    elif result == False:
                        display_message(screen, "TIE GAME!")

                    game_running = False

        for event in pygame.event.get():

            # Did the user hit a key?

            if event.type == KEYDOWN:

                # Was it the Escape key? If so, stop the loop.

                if event.key == K_ESCAPE:
                    running = False

            # Did the user click the window close button? If so, stop the loop.

            elif event.type == QUIT:

                running = False

            elif event.type == MOUSEBUTTONDOWN and is_players_move and game_running:
                is_players_move = False
                board, piece = mouse_pos_to_board_and_piece(pygame.mouse.get_pos())

                test_board = minimax_node.board.copy_board()

                test_board.move(board, piece)

                # In order to avoid restarting the node tree every time, find the node among current_game_node.children with
                # the board that matches the player's move
                found_board = False

                if len(minimax_node.children) == 0:
                    minimax_node.add_children()

                for i in minimax_node.children:

                    # If the board matches, use this as the new board
                    if i.board.board == test_board.board:
                        minimax_node = i
                        found_board = True

                if not found_board:
                    print(
                        "Your move was not found as a valid move. Are you sure you entered it correctly?"
                    )
                    print(minimax_node.children)
                    is_players_move = True
                    continue

                # draw_move(screen, board, piece, "O", font)
                game.move(board, piece)

                screen.fill((255, 255, 255))

                print(game.to_move)
                print(game)

                draw_game(screen, game)

                result = game.game_result
                if result is not None:
                    if result == 1:
                        display_message(screen, "X WINS!")
                    elif result == -1:
                        display_message(screen, "O WINS!")
                    elif result == False:
                        display_message(screen, "TIE GAME!")

                    game_running = False
                    continue

                minimax_results, p = mcts.minimax_search_async(
                    minimax_node, DEPTH, not HUMAN_PLAY_AS_O
                )

    # for event in pygame.event.get():
    #
    #     if event.type == MOUSEBUTTONDOWN:
    #
