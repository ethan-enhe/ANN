#include <bits/stdc++.h>
#include "../include/microdnn.h"
using namespace std;
int main() {
    sequential net;
    net.add(make_shared<linear>(2, 10));
    net.add(make_shared<relu>());
    net.add(make_shared<linear>(10, 10));
    net.add(make_shared<relu>());
    net.add(make_shared<linear>(10, 1));
    net.add(make_shared<sigmoid>());

    batch dat;
    for (int i = 1; i <= 10000; i++) {
        float x = rd(0, 1), y = rd(0, 1);
        float label = 0;
        if (x * x + y * y >= 0.5 * 0.5 && x * x + y * y <= 1 * 1) label = 1;
        dat.first.push_back(make_vec({x, y}));
        dat.second.push_back(make_vec({label}));
    }
    data_set divided_dat(dat);

    adam opt;
    upd(opt, divided_dat, net, 32, 10000, chk_2, "model.txt");

    return 0;
}
