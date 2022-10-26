#include <bits/stdc++.h>

#include "../include/net.h"

using namespace std;

#include "board.h"
BP net;
vector<pair<vector<double>, vector<double>>> tdata, _tdata;
Reversi_Board b;
int main() {
    /* fstream fin; */
    /* fin.open("reversi.txt", ios::in); */
    /* fin >> net; */
    /* fin.close(); */
    /* string data, x, y; */
    /* cin >> data >> x >> y; */
    /* data+=x+y; */
    /* Reversi_Board b; */
    /* int delt = (x[1] - '0') * 10 + (x[2] - '0'); */
    /* if (x[0] == '-') delt = -delt; */
    /* int cur = -1; */
    /* while (b.win() == -2) { */
    /*     cur = -cur; */
    /*     b.putchess(data[1] - '0', data[0] - 'a' + 1, cur); */
    /*     b.raw_prt(); */
    /*     data = data.substr(2); */
    /*     auto r = net.run(b.vectorize(1)); */
    /*     cout << r.front() << endl; */
    /* } */
    /* cout<<b.win(); */

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
            /* b.raw_prt(); */
            /* cerr << data.substr(ind, 2) << endl; */
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
            if (data.size()-ind<=20) {
                if (ri(0, 5) == 0) {
                    _tdata.push_back({b.vectorize(1), {delt > 0 ? 0.9 : (delt < 0 ? 0.1 : 0.5)}});
                    _tdata.push_back({b.vectorize(-1), {delt > 0 ? 0.1 : (delt < 0 ? 0.9 : 0.5)}});
                } else {
                    tdata.push_back({b.vectorize(1), {delt > 0 ? 0.9 : (delt < 0 ? 0.1 : 0.5)}});
                    tdata.push_back({b.vectorize(-1), {delt > 0 ? 0.1 : (delt < 0 ? 0.9 : 0.5)}});
                }
            }
        }
        /* break; */
    }
    fin.close();

    /* fin.open("reversi.txt", ios::in); */
    /* fin >> net; */
    /* fin.close(); */

    net.init({64, 200, 200, 200, 1});
    net.train_all(tdata, _tdata, 0.004, 1e6,10000);
    fstream fout;
    fout.open("reversi.txt", ios::out);
    fout << net;
    fout.close();
    return 0;
}
