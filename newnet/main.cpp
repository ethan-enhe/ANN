#include <bits/stdc++.h>

#include "../include/newnet.h"
#include "../include/rnd.h"

using namespace std;
int main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    ANN FCN;
    FCN.add<same, int>(2);
    FCN.add<hardswish, int>(20);
    FCN.add<hardswish, int>(20);
    //FCN.add<hardswish, int>(20);
    FCN.add<sigmoid, int>(1);
    dataset train, test;
    for (int i = 1; i <= 10000; i++) {
        double x = rd(-1, 1), y = rd(-1, 1);
        VectorXd in(2), out(1);
        in << x, y;
        out << (x * x + y * y <= 0.5 && x * x + y * y >= 0.3 ? 1 : 0);
        if (ri(0, 5) == 0)
            test.push_back({in, out});
        else
            train.push_back({in, out});
    }
    FCN.adam(train, test, 8, 100000, 10000,crossentropy);
    //FCN.sgd(train, test, 8, 100000, 10000, [](int x) { return 1. /8./ (1. + x * 0.005); },crossentropy); 
    while (1) {
        double x, y;
        cin >> x >> y;
        VectorXd in(2), out(1);
        in << x, y;
        out = FCN.forward(in);
        cout << out << endl;
    }
    return 0;
}
