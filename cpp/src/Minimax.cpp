#include "Minimax.h"
#include "GameState.h"
#include <bitset>
#include <math.h>

using namespace std;

float evaluate(GameState board) {
    /**
     * Evaluates the given position.
     * @return The evaluation, Positive indicates advantage to X, negative indicates advantage to O
     */
    float miniboardEvalsX[9], miniboardEvalsO[9];
    bitset<20> (&position)[9] = board.board;

    float finalEval = 0;

    int status = board.getStatus();
    if (status == 1) { // X wins
        return numeric_limits<float>::infinity();
    } else if (status == 2) { // O wins
        return -1 * numeric_limits<float>::infinity();
    } else if (status == 3) { // Tie game
        return 0;
    }

    for (int i = 0; i < 9; i++) {
        miniboardEvalsX[i] = miniboardEvalOneSide(position[i], 1);
        miniboardEvalsO[i] = miniboardEvalOneSide(position[i], 2);
    }

    significances sigs;
    sigs = calcSignificances(position, miniboardEvalsX, miniboardEvalsO);
    
    for (int i = 0; i < 9; i++)
    {
        finalEval += (miniboardEvalsX[i] * sigs.sigsX[i]) - (miniboardEvalsO[i] * sigs.sigsO[i]);
    }
    

    return finalEval;
}


float miniboardEvalOneSide(bitset<20> miniboard, int side) {
    /**
     * Evaluates a single miniboard for one side.
     * 
     * eval = (c1 * (w ^ 0.5)) + c2 * r)
     * 
     * Where:
     * c1, c2 - constants
     * w - spaces that are winning if taken
     * r - rows with only one spot taken (ie win in two moves)
     * 
     * cw - the value of an already won board
     * cl - the value of a lost board (should be <=0)
     * ct - the value of a tied board
     * 
     * @param side 1 to evaluate for X; 2 to evaluate for O
     */

    int r = 0, w = 0;


    // Check for win/loss

    int result = getMiniboardResults(miniboard);
    if (result == side) {
        return cw;
    // Check if the other side won
    } else if (result == 2 / side) {
        return cl;
    } else if (result == 3) {
        return ct;
    }

    // Calculate w and r
    int index = 0;

    // The amount to move from location to get to the corresponding bit for the other side
    int sideOffset = 0, posOffset = 0;
    if (side == 1) {
        sideOffset = 1;
    } else {
        sideOffset = 0;
        posOffset = 1;
    }

    for (int location = 1 + side; location < 20; location += 2) {
        
        if (!miniboard[location]) {
            // Check each winning possibility
            for (int i = 0; i < 4; i++) {
                if (winningPossibilitiesLocations[index][i][0] == -1) {
                    break;
                }

                if (miniboard[winningPossibilitiesLocations[index][i][0] + posOffset] && miniboard[winningPossibilitiesLocations[index][i][1] + posOffset]) {
                    w++;
                    break;
                }
            }
        } else {
            for (int i = 0; i < 4; i++) {
                // If this spot is taken by us and the other two are empty, this is a win-in-two index
                if (winningPossibilitiesLocations[index][i][0] == -1) {
                    break;
                }

                if (!miniboard[winningPossibilitiesLocations[index][i][0] + posOffset] &&
                        !miniboard[winningPossibilitiesLocations[index][i][0] + sideOffset] &&
                        !miniboard[winningPossibilitiesLocations[index][i][1] + posOffset] &&
                        !miniboard[winningPossibilitiesLocations[index][i][1] + sideOffset]) {
                    r++;
                }
            }
        }

        index += 1;
    }

    return c1 * sqrt(w) + c2 * r;
}

significances calcSignificances(bitset<20> fullBoard[9], float evaluationsX[9], float evaluationsY[9]) {
    /**
     * Calculate the significances of each miniboard based on the evaluations given.
     */
    significances result;

    int result1, result2;
    int winCoords1, winCoords2;

    // Loop through each miniboard
    for (int i = 0; i < 9; i++)
    {
        
        if (fullBoard[i][0] && fullBoard[i][1]) {

            result.sigsO[i] = tieSig;
            result.sigsX[i] = tieSig;
        } else if (fullBoard[i][0]) {

            result.sigsX[i] = wonSig;
            result.sigsO[i] = lostSig;
        } else if (fullBoard[i][1]) {

            result.sigsO[i] = wonSig;
            result.sigsX[i] = lostSig;

        } else {
            result.sigsX[i] = 0;
            result.sigsO[i] = 0;
            // Loop through each winning possibility
            for (int winIndex = 0; winIndex < 4; winIndex++) {
                winCoords1 = winningPossibilities[i][winIndex][0];
                winCoords2 = winningPossibilities[i][winIndex][1];

                // If either board is already won for the other side (or tied), sig is zero
                if (!fullBoard[winCoords1][1] && !fullBoard[winCoords2][1]) {
                    result1 = evaluationsX[winCoords1];
                    result2 = evaluationsX[winCoords2];
                    result.sigsX[i] += result1 + result2;
                }

                if (!fullBoard[winCoords1][0] && !fullBoard[winCoords2][0]) {
                    result1 = evaluationsY[winCoords1];
                    result2 = evaluationsY[winCoords2];
                    result.sigsO[i] += result1 + result2;
                }

            }
        }
    }
    
    return result;
};

Node::Node (GameState currentBoard, int currentDepth){
    board = currentBoard;
    depth = currentDepth;
}

void Node::addChildren() {
    /**
     * Add all possible moves as children. Checks if children have already been added and will not add again.
     */
    if (!hasChildren) {
        for (GameState i : board.allPossibleMoves()) {
            children.push_back(Node(i, depth + 1));
        }
        hasChildren = true;
    }
}

float minimax(Node (&node), int depth, float alpha, float beta, bool maximizingPlayer) {
    /**
     * Calculates the evaluation of the given board to the given depth.
     * Note that updateMiniboardStatus() or updateSignleMiniboardStatus() must be called before this function.
     */

    float bestEval, newEval;

    // Check if depth is reached or game is over
    if (depth <= 0 || node.board.getStatus() != 0) {
        bestEval = evaluate(node.board);

        if (isinf(bestEval)) {
            // Save the depth if the evaluation is infinite
            node.infDepth = node.depth;
        }

        return bestEval;
    }

    // Init with worst outcome, so anything else is always better
    if (maximizingPlayer)
        bestEval = -1 * numeric_limits<float>::infinity();
    else
        bestEval = numeric_limits<float>::infinity();

    node.addChildren();

    for (Node i : node.children) {
        newEval = minimax(i, depth - 1, alpha, beta, !maximizingPlayer);

        if (maximizingPlayer) {
            // Get the highest evaluation
            bestEval = (newEval > bestEval) ? newEval : bestEval;

            // If the position is lost, the best option is the one furthest from game over
            if (newEval == -1 * numeric_limits<float>::infinity()) {
                if (node.infDepth == -1 || node.infDepth < i.infDepth) {
                    node.infDepth = i.infDepth;
                }
            }

            // If the position is won, the best option is the one closest from game over
            else if (newEval == numeric_limits<float>::infinity()) {
                if (node.infDepth == -1 || node.infDepth > i.infDepth) {
                    node.infDepth = i.infDepth;
                }
            }

            alpha = (alpha > newEval) ? alpha : newEval;

            // Prune the position
            if (beta <= alpha) {
                break;
            }
        } else {
            // Get the lowest evaluation
            bestEval = (newEval < bestEval) ? newEval : bestEval;

            // If the position is lost, the best option is the one furthest from game over
            if (newEval == numeric_limits<float>::infinity()) {
                if (node.infDepth == -1 || node.infDepth < i.infDepth) {
                    node.infDepth = i.infDepth;
                }
            }

            // If the position is won, the best option is the one closest from game over
            else if (newEval == -1 * numeric_limits<float>::infinity()) {
                if (node.infDepth == -1 || node.infDepth > i.infDepth) {
                    node.infDepth = i.infDepth;
                }
            }

            beta = (beta < newEval) ? beta : newEval;

            // Prune the position
            if (beta <= alpha) {
                break;
            }
        }
    }

    return bestEval;
};

GameState minimaxSearch(GameState position, int depth, bool playAsX) {
    Node start = Node(position, 0);

    start.addChildren();

    float bestEval = numeric_limits<float>::infinity() * -1;
    float newEval;
    Node bestMove = start.children[0];
    int evalMultiplier = (playAsX) ? 1 : -1;

    for (Node i : start.children) {
        newEval = minimax(i, depth - 1, -1 * numeric_limits<float>::infinity(), numeric_limits<float>::infinity(), !playAsX);

        newEval *= evalMultiplier;

        if (newEval > bestEval) {
            bestEval = newEval;
            bestMove = i;
        }

    }

    return bestMove.board;
};
