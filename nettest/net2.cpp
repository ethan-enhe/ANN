
#include "../include/optimize.h"
#include "../include/net2.h"


using namespace std;
int main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    layer_seq FCN;
    FCN.add(make_shared<linear>(2, 20));
    FCN.add(make_shared<hardswish>());
    FCN.add(make_shared<linear>(20, 20));
    FCN.add(make_shared<hardswish>());
    FCN.add(make_shared<linear>(20, 1));
    FCN.add(make_shared<sigmoid>());
    batch data;
    for (int i = 1; i <= 10000; i++) {
        VectorXd in(2), out(1);
        /* double x = ri(0, 1), y = ri(0, 1); */
        double x = rd(-1, 1), y = rd(-1, 1);
        in << x, y;
        out << (x * x + y * y <= 0.5 && x * x + y * y >= 0.3 ? 1 : 0);
        /* out << (int(x) ^ int(y)); */
        data.first.push_back(in);
        data.second.push_back(out);
    }
    data_set sliced_data(data);
    sgd(
        sliced_data, FCN, 32, 100000, [](int x) { return 1. / 32. / (1. + x * 0.005); }, variance);

    while (1) {
        double x, y;
        cin >> x >> y;
        VectorXd in(2), out(1);
        in << x, y;
        out = FCN.forward(vec_batch{in})[0];
        cout << out << endl;
    }
    return 0;
}
