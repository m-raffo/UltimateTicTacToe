#include <iostream>
#include <bitset>
#include <vector>

using namespace std;

// Masks for winning positions
const bitset<20> winningPosX[] = {
    0b00000000000001010100,
    0b00000001010100000000,
    0b01010100000000000000,
    0b00000100000100000100,
    0b00010000010000010000,
    0b01000001000001000000,
    0b01000000010000000100,
    0b00000100010001000000,
};

const bitset<20> winningPosO[] = {
    0b00000000000010101000,
    0b00000010101000000000,
    0b10101000000000000000,
    0b00001000001000001000,
    0b00100000100000100000,
    0b10000010000010000000,
    0b10000000100000001000,
    0b00001000100010000000,
};

class GameState {
    
    private:

    /**
     * Info - stores information about the GameState that is not stored on the board
     * Bits:
     * 0-3: Required board to move on
     * 4: Is there a required board (1=yes, 0=no)
     * 5: Player to move (1=X, 0=O)
     */ 
    char info;

    std::bitset<20> board[9];

    int checkMiniboardResults(bitset<20> miniboard) {
        /**
         * Evaluates the given miniboard to check for wins.
         * 0: Ongoing game
         * 1: X win
         * 2: O win
         * 3: Tie
         */
        bitset<20> posToCheck;

        // If the position is empty, return 0
        if (miniboard == posToCheck) {
            return 0;
        }

        bool emptySpace = false;
        // Check if the position is completely filled
        for (int j = 0; j <= 8; j++) {
            int location = 2 + (j * 2);
            if (!miniboard[location] && !miniboard[location + 1]) {
                emptySpace = true;
                break;
            }
        }

        // If there are no empty spaces, mark the position as a tie
        if (!emptySpace) {
            return 3;
        }

        // Check each winning possibility
        for (int j = 0; j <= 7; j++) {
            // Match the position with the winning mask
            posToCheck = winningPosX[j];
            posToCheck &= miniboard;

            // If the position matches the winning mask, mark it as a win
            if (posToCheck == winningPosX[j]) {
                return 1;
            }

            // Same for O
            posToCheck = winningPosO[j];
            posToCheck &= miniboard;

            if (posToCheck == winningPosO[j]) {
                return 2;
            }
        }

        return 0;
    }

    public:
    void setToMove(int m) {
        /**
         * Sets the player to move
         * 1:X
         * 2:O
         */
        if (m) {
            // Set bit 5 of info to 1
            info |= 1 << 5;
        } else if (!m) {
            // Set bit 5 of info to 0
            info &= ~(1 << 5);
        }
    }

    int getToMove() {
        /**
         * Sets the player to move
         * 1:X
         * 2:O
         */
        // Check bit 5 of info is set
        char toMove = 1 << 5;
        toMove |= info;
        if (toMove) {
            return 1;
        } else {
            return 2;
        }
    }

    void setRequiredBoard(int requiredBoard) {
        /**
         * Sets the required board for the next move.
         * 0-8 = that board to move on
         * -1 = no required board
         */

        // set bit 4 to 0 if there is no required board
        if (requiredBoard == -1) {
            info &= ~(1 << 4);
        } else {
            // set bit 4 to 1 b/c there is a required board
            info |= 1 << 4;

            // Set bits 0-3 to 0 while leaving everything else untouched
            info &= ~(0) << 4;

            // Set bits 0-3 to the correct required board
            info |= requiredBoard;
        }
    }

    int getRequiredBoard() {
        /**
         * Gets the required board for the next move.
         * 0-8: That board must be moved on
         * -1: No required board
         */

        // Check if there is a required board
        char requiredBoard = 1 << 4;
        requiredBoard |= info;

        if (requiredBoard) {
            requiredBoard = 15;  // Sets the first 4 bits to 1

            requiredBoard &= info;
            return requiredBoard;
        } else {
            return -1;
        }
    }

    int getPosition(int boardLocation, int pieceLocation) {
        /**
         * Gets the piece in the specified location in the board.
         * 
         * @param boardLocation The board (from 0 to 8) to get
         * @param pieceLocation The piece (fromr 0 to 8) to get
         * @return 0 if the position is empty, 1 if the position is claimed by X, and 2 if the position is claimed by O
         */

        // Miniboards are 20 bits longs
        // Spots are 2 bits long
        // The first two bits of each miniboard are for storing the results of the miniboard to avoid recalculation if possible
        int location = (2 * pieceLocation) + 2;

        if (board[boardLocation][location]) {
            return 1;
        } else if (board[boardLocation][location + 1]) {
            return 2;
        } else {
            return 0;
        }
        
    }

    void setPosition(int boardLocation, int pieceLocation, int piece) {
        /**
         * Sets the specificed location in the board to the given piece.
         * 
         * @param boardLocation The board (from 0 to 8) to get
         * @param pieceLocation The piece (fromr 0 to 8) to get
         * @param piece The piece to set. 1 for X; 2 for O; 0 for empty
         * @return void
         */

        int location = (2 * pieceLocation) + 2;

        if (piece == 0) {
            board[boardLocation][location] = 0;
            board[boardLocation][location + 1] = 0;
        } else if (piece == 1) {
            board[boardLocation][location] = 1;
            board[boardLocation][location + 1] = 0;
        } else {
            board[boardLocation][location] = 0;
            board[boardLocation][location + 1] = 1;
        }
    }

    void move(int boardLoaction, int pieceLocation) {
        /**
         * Performs the specificed move on the board, moving the piece whose turn it is.
         */

        setPosition(boardLoaction, pieceLocation, getToMove());
    }

    void updateMiniboardStatus() {
        /**
         * Updates the game statuses of all the miniboards, checking to see if any of them are won.
         * If a position is won for both O and X and not already marked, it will be marked as a win for X.
         */

        // Loop through each miniboard
        for (int i = 0; i <= 8; i++) {

            // Check if already marked as a finished position
            if (board[i][0] || board[i][1]) {
                continue;
            }

            int result = checkMiniboardResults(board[i]);

            // Tie
            if (!result) {
                continue;
            }

            // X Wins
            else if (result == 1) {
                board[i][0] = 1;
            }

            // O Wins
            else if (result == 2) {
                board[i][1] = 1;
            }

            // Tie
            else {
                board[i][0] = 1;
                board[i][1] = 1;
            }
        }

    }

    int getBoardStatus(int boardLocation) {
        /**
         * Gets the status of the given miniboard.
         * Important: GameState.updateMiniboardStatus() MUST be called before this function to ensure correct results.
         * 
         * 0: Ongoing game
         * 1: X wins
         * 2: O wins
         * 3: Tie
         * @param boardLocation the board to check from 0 to 8
         * @return the status
         */

        // Tie
        if (board[boardLocation][0] && board[boardLocation][1]) {
            return 3;
        }

        // X wins
        else if (board[boardLocation][0]) {
            return 1;
        }

        // O wins
        else if (board[boardLocation][1]) {
            return 2;
        }

        // Ongoing game
        else {
            return 0;
        }
    }

    int getStatus() {
        /**
         * Gets the status of the entire game.
         * Important: GameState.updateMiniboardStatus() MUST be called before this function to ensure correct results.
         * 
         * 0: Ongoing game
         * 1: X wins
         * 2: O wins
         * 3: Tie
         * @return the status
         */
        bitset<20> boardResults;

        for (int i = 0; i <= 8; i++) {
            // The location in board results to store this result
            int location = 2 + (i * 2);
            int result = getBoardStatus(i);

            switch (result) {
            // X win
            case 1:
                boardResults[location] = 1;
                break;

            // O win
            case 2:
                boardResults[location + 1] = 1;
                break;

            // Tie
            case 3:
                boardResults[location] = 1;
                boardResults[location + 1] = 1;
            
            // Ongoing game
            default:
                break;
            }
        }

        return checkMiniboardResults(boardResults);

    }

    vector<GameState> allPossibleMoves() {
        vector<GameState> allMoves;

        
    }
};

class Rectangle {
    public:
        int width, height;

        int area(void) {
            return width * height;
        };
};

int main() {
    GameState myboard;


    myboard.updateMiniboardStatus();
    cout << myboard.getStatus() << '\n';

    myboard.setPosition(0, 0, 1);
    myboard.setPosition(0, 1, 1);
    myboard.setPosition(0, 2, 1);

    myboard.setPosition(1, 0, 1);
    myboard.setPosition(1, 1, 1);
    myboard.setPosition(1, 2, 1);

    myboard.setPosition(2, 0, 1);
    myboard.setPosition(2, 1, 1);
    myboard.setPosition(2, 2, 1);

    myboard.updateMiniboardStatus();
    cout << myboard.getStatus() << '\n';


    return 0;
}
