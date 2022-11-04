#include "../include/net2.h"

#include <bits/stdc++.h>

#include "board.h"

using namespace std;

layer_seq net;
batch tdata;
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
            if (data.size() - ind < 2 && ri(0, 6) == 0) {
                VectorXd in = make_vec(b.vectorize(1)), out(1), _out(1);
                out << (delt > 0 ? 1 : (delt < 0 ? 0 : 0.5));
                _out << (delt > 0 ? 0 : (delt < 0 ? 1 : 0.5));
                tdata.first.push_back(in);
                tdata.second.push_back(out);
                tdata.first.push_back(-in);
                tdata.second.push_back(_out);
            }
        }
    }
    fin.close();

    net.add(make_shared<linear>(64, 128));
    net.add(make_shared<mish>());
    net.add(make_shared<batchnorm>(128));

    net.add(make_shared<linear>(128, 128));
    net.add(make_shared<mish>());
    net.add(make_shared<batchnorm>(128));

    net.add(make_shared<linear>(128, 128));
    net.add(make_shared<mish>());
    net.add(make_shared<batchnorm>(128));

    net.add(make_shared<linear>(128, 128));
    net.add(make_shared<mish>());
    net.add(make_shared<batchnorm>(128));

    net.add(make_shared<linear>(128, 1));
    net.add(make_shared<same>());

    data_set sliced(tdata);
    adam(sliced, net, 64, 50000, sqrtvariance, "test.txt");

    /* net.readf("./test.txt"); */
    /*     net.set_train_mode(0); */
    /*     string a, tmp, c; */
    /*     while (1) { */
    /*         cin >> a >> tmp >> c; */
    /*         string data = a + tmp + c; */
    /*         int last = -1, ind = 0; */
    /*         b.init(); */
    /*         int step = 0; */
    /*         while (b.win() == -2) { */
    /*             ++step; */
    /*             last = -last; */
    /*             bool canput = 0; */
    /*             int xc = data[ind + 1] - '0'; */
    /*             int yc = data[ind] - 'a' + 1; */
    /*             for (int i = 1; i <= 8; i++) */
    /*                 for (int j = 1; j <= 8; j++) */
    /*                     if (!b.board[i][j] && b.eat(0, xc, yc, last)) { */
    /*                         canput = 1; */
    /*                         break; */
    /*                     } */
    /*             if (canput) { */
    /*                 ind += 2; */
    /*                 b.putchess(xc, yc, last); */
    /*                 /1* b.raw_prt(); *1/ */
    /*                 cout << "step: " << step << endl; */
    /*                 cout << net.forward({make_vec(b.vectorize(1))})[0](0) * 200000 - 100000 << endl; */
    /*                 cout << b.assess(1) << endl; */
    /*             } */
    /*         } */
    /*     } */
    return 0;
}
