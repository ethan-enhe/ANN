#include <bits/stdc++.h>

#include "../include/newnet.h"
#include "board.h"

using namespace std;

ANN net;
dataset tdata, _tdata;
Reversi_Board b;
int main() {
    fstream fin;
    fin.open("./games.txt", ios::in);
    string data;
    int cnt = 0;
    while (fin >> data) {
        ++cnt;
        if (cnt % 10000 == 0) cerr << cnt << endl;

        int last = -1, ind = 0;
        b.init();
        while (b.win() == -2) {
            last = -last;
            bool canput = 0;
            int xc = data[ind + 1] - '0';
            int yc = data[ind] - 'A' + 1;
            ind += 2;
            for (int i = 1; i <= 8; i++)
                for (int j = 1; j <= 8; j++)
                    if (!b.board[i][j] && b.eat(0, xc, yc, last)) {
                        canput = 1;
                        break;
                    }
            if (!canput) last = -last;
            b.putchess(xc, yc, last);
        }
        last = -1, ind = 0;
        int delt = b.win();
        b.init();
        while (b.win() == -2) {
            last = -last;
            bool canput = 0;
            int xc = data[ind + 1] - '0';
            int yc = data[ind] - 'A' + 1;
            ind += 2;
            for (int i = 1; i <= 8; i++)
                for (int j = 1; j <= 8; j++)
                    if (!b.board[i][j] && b.eat(0, xc, yc, last)) {
                        canput = 1;
                        break;
                    }
            if (!canput) last = -last;
            b.putchess(xc, yc, last);
            if (data.size() - ind <5 && ri(0, 10) == 0) {
                VectorXd in = std2eigen(b.vectorize(1)), out(1), _out(1);
                out << (delt > 0 ? 1 : (delt < 0 ? 0 : 0.5));
                _out << (delt > 0 ? 0 : (delt < 0 ? 1 : 0.5));
                if (ri(0, 5) == 0) {
                    _tdata.push_back({in, out});
                    _tdata.push_back({-in, _out});
                } else {
                    tdata.push_back({in, out});
                    tdata.push_back({-in, _out});
                }
            }
        }
        /* break; */
    }
    fin.close();

    net.add<same, int>(64);
   net.add<hardswish, int>(128);
    net.add<hardswish, int>(128);
    net.add<sigmoid, int>(1);
    //net.sgd(tdata, _tdata, 8, 100000, 1000, [](int i) { return 1. / 8 / 1 + 0.005 * i; },variance);
    net.adam(tdata, _tdata, 8, 100000, 1000, variance);
    /* fstream fout; */
    /* fout.open("reversi.txt", ios::out); */
    /* fout << net; */
    /* fout.close(); */
    return 0;
}
