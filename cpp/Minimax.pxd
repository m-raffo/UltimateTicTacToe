cdef extern from "src/Minimax.cpp":
    pass

cdef extern from "src/GameState.cpp":
    pass

cdef extern from "include/GameState.h":
    cdef cppclass GameState:
        GameState() except +
        void move(int, int)
        GameState getCopy()
        int getStatus()

        int getPosition(int, int)

    cdef struct boardCoords:
        char board, piece


cdef extern from "include/Minimax.h":
    cdef cppclass Node:
        Node() except +
        Node(GameState, int) except +
        int infDepth, depth
        GameState board

    cdef boardCoords minimaxSearchMove(GameState, int, bool)

