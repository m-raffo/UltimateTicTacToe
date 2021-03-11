#include <bitset>
#include <vector>
#include <iostream>
#include <GameState.h>

using namespace std;

int main() {
    GameState myboard;

    myboard.move(4, 5);
    myboard.displayGame();

    vector<GameState> allMoves = myboard.allPossibleMoves();

    cout << "ALL POSSIBLE MOVES!!!!!" << '\n';
    for (int i = 0; i < allMoves.size(); i++) {
        allMoves[i].displayGame();
    }


    return 0;
}
