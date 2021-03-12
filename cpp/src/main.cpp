#include <bitset>
#include <vector>
#include <iostream>
#include <GameState.h>
#include <Minimax.h>
#include <chrono>

using namespace std;

int main() {
    GameState myboard;

    myboard.move(4, 5);
    myboard.move(5, 0);
    myboard.move(0, 1);
    myboard.move(1, 1);
    myboard.move(1, 8);
    myboard.move(8, 4);
    myboard.move(4, 3);
    myboard.move(3, 0);
    myboard.move(0, 0);
    myboard.move(0, 6);
    myboard.move(6, 3);
    myboard.move(3, 1);
    // myboard.move(1, 2);

    myboard.displayGame();


    auto start = chrono::high_resolution_clock::now();



    // vector<GameState> allMoves = myboard.allPossibleMoves();
    // cout << "IN BETWEEN \n";
    float evaluation = evaluate(myboard);

    GameState bestMove = minimaxSearch(myboard, 8, true);


    auto stop = chrono::high_resolution_clock::now();

    bestMove.displayGame();

    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);

    cout << "TIME TAKEN ";
    cout << duration.count() << " milliseconds\n\n\n";

    return 0;
}
