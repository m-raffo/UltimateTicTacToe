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
    myboard.move(1, 2);

    myboard.displayGame();


    cout << "um hello\n";

    auto start = chrono::high_resolution_clock::now();



    // vector<GameState> allMoves = myboard.allPossibleMoves();
    // cout << "IN BETWEEN \n";
    float evaluation = evaluate(myboard.board);

    auto stop = chrono::high_resolution_clock::now();


    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);

    cout << "TIME TAKEN";
    cout << duration.count() << " microseconds\n\n\n";

    return 0;
}
