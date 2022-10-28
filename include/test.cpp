#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2,bmi,bmi2,lzcnt,popcnt")
#include "net2.h"
using namespace std;
layer_seq net;
int main() {
    mat x(2, 2), y(2, 2);
    x << 1, 2, 3, 4;
    y << 1, 2, 3, 4;
    cout << (x.array() / y.array());
    /* net.add(make_shared<batchnorm>(1)); */
    /* net.add(make_shared<same>()); */
    /* cout << net.shape(); */
    /* vec_batch in, out, ans, grad; */
    /* in = { */
    /*     make_vec({1}), */
    /*     make_vec({0.5}), */
    /*     make_vec({0}), */
    /* }; */
    /* ans = { */
    /*     make_vec({0}), */
    /*     make_vec({0}), */
    /*     make_vec({1}), */
    /* }; */
    /* out = net.forward(in); */
    /* grad = net.backward(ans); */
    /* double err = variance(out, ans) * 3; */
    /* cout << endl << err; */

    /* /1* for (auto &x : out) cout << x << endl; *1/ */
    /* /1* in = { *1/ */
    /* /1*     make_vec({1.1}), *1/ */
    /* /1*     make_vec({0.5}), *1/ */
    /* /1*     make_vec({0}), *1/ */
    /* /1* }; *1/ */

    /* /1* for (auto &x : out) cout << x << endl; *1/ */
    /* net.set_train_mod(0); */

    /* out = net.forward(in); */
    /* err = variance(out, ans) * 3; */
    /* cout << endl << err; */

    /* cout << endl << (variance(out, ans) * 3 - err) / grad[0](0) / 0.01 << endl << endl; */

    /* cout<<net.backward(ans)[0]<<endl; */
    /* cout<<crossentropy_2(out, ans)<<endl; */
    /* in = { */
    /*     make_vec({1, -0.1}), */
    /*     make_vec({0, 1}), */
    /*     make_vec({1, 1}), */
    /*     make_vec({0, 0}), */
    /* }; */
    /* out=net.forward(in); */
    /* cout<<crossentropy_2(out, ans)<<endl; */

    /*     for (int i = 0; i <= 100000; i++) { */
    /*         out = net.forward(in); */
    /*         grad = net.backward(ans); */
    /*         cout << variance(out, ans) << endl; */
    /*         net.upd(0.01); */
    /*         /1* net.upd(0.25 / (1. + 0.005 * i)); *1/ */
    /*     } */
    return 0;
}
