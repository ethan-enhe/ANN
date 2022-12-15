#include "../include/net2.h"

#include <bits/stdc++.h>

#include <memory>

using namespace std;
int main() {
    sequential FCN;

    // FCN.add(make_shared<conv>(1, 1, 2, 2, 2, 2));
    FCN.add(make_shared<batchnorm>(1));
    // FCN.add(make_shared<maxpool2x2>(1, 2, 2));
    FCN.add(make_shared<same>());
    vec_batch in1 = {make_vec({1})};
    vec_batch out = {make_vec({2})};
    FCN.readf("tmp");
    cout << FCN.forward(in1)[0] << endl;
    FCN.writef("tmp");
    // layer* x = FCN.layers[1].get();
    // conv* y = (conv*)x;
    // cout << y->grad_kernel << endl << endl;
    // y->kernel(0, 0, 0, 0) += 0.1;
    // FCN.forward(in1);
    // cout << FCN.out()[0] << endl << endl;
    // FCN.forward(in2);
    // cout << FCN.out()[0] << endl;
    //
    // FCN.add(make_shared<linear>(2, 128));
    // FCN.add(make_shared<hardswish>());
    // FCN.add(make_shared<batchnorm>(128));
    //
    // FCN.add(make_shared<linear>(128, 128));
    // FCN.add(make_shared<hardswish>());
    //
    // FCN.add(make_shared<linear>(128, 1));
    // FCN.add(make_shared<sigmoid>());
    //
    // batch data;
    // for (int i = 1; i <= 10000; i++) {
    //     vec in(2), out(1);
    //     double x = rd(-1, 1), y = rd(-1, 1);
    //     in << x, y;
    //     out << (x * x + y * y <= 0.5 && x * x + y * y >= 0.3 ? 1 : 0);
    //     //
    //     // VectorXd in(64), out(1);
    //     // for (auto &x : in) x = rd(-1, 1);
    //     // out(0) = in.sum() > 0;
    //
    //     data.first.push_back(in);
    //     data.second.push_back(out);
    // }
    // data_set sliced(data);
    // adam ada;
    // nesterov nes(0.01, 0.9, 0.01);
    // upd(ada, sliced, FCN, 32, 100000, crossentropy_2);
    // /* adam(sliced, FCN, 32, 10000, sqrtvariance); */
    // /* data_set sliced_data(data); */
    // /* /1* sgd( *1/ */
    // /* /1* sliced_data, FCN, 64, 10000, [](int x) { return 1. / 64. / (1. + x * 0.005); }, chk_k); *1/ */
    // /* adam(sliced_data, FCN, 128, 100000, chk_2); */
    // /* FCN.set_train_mode(0); */
    // /* while (1) { */
    // /*     double x, y; */
    // /*     cin >> x >> y; */
    // /*     VectorXd in(2), out(1); */
    // /*     in << x, y; */
    // /*     out = FCN.forward(vec_batch{in})[0]; */
    // /*     cout << out << endl; */
    // /* } */
    return 0;
}
// 0.049 0.049 0.063 0.048
// 0.059 0.06 0.05 0.062
