# distutils: language = c++

from test cimport Node

cdef class PyGameState:
    cdef GameState c_gamestate


    def __cinit__(self):
        self.c_gamestate = GameState()

    def get_copy(self):
        new_copy = PyGameState()
        new_copy.c_gamestate = self.c_gamestate.getCopy()

        return new_copy

    def move(self, board, piece):
        self.c_gamestate.move(board, piece)

    def get_status(self, ):
        return self.c_gamestate.getStatus()

    def get_position(self, board, piece):
        return self.c_gamestate.getPosition(board, piece)

    def minimax_search_move(self, depth, playAsX):
        cdef boardCoords nextMove
        
        nextMove = minimaxSearchMove(self.c_gamestate, depth, playAsX)

        return [nextMove.board, nextMove.piece]
