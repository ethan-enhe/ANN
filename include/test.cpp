#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2,bmi,bmi2,lzcnt,popcnt")
#include "net2.h"
using namespace std;
layer_seq net;
int main() {
    net.add(make_shared<linear>(2, 10));
    net.add(make_shared<relu>());
    net.add(make_shared<linear>(10, 10));
    net.add(make_shared<relu>());
    net.add(make_shared<linear>(10, 10));
    net.add(make_shared<hardswish>());
    net.add(make_shared<linear>(10, 1));
    net.add(make_shared<sigmoid>());
    cout << net.shape();
    vec_batch in, out, ans, grad;
    in = {
        make_vec({1, 0}),
        make_vec({0, 1}),
        make_vec({1, 1}),
        make_vec({0, 0}),
    };
    ans = {
        make_vec({1}),
        make_vec({1}),
        make_vec({0}),
        make_vec({0}),
    };
    for (int i = 0; i <= 100000; i++) {
        out = net.forward(in);
        grad = net.backward(ans);
        cout << crossentropy_2(out, ans) << endl;
        /* net.upd(0.01); */
        net.upd(0.25 / (1. + 0.005 * i));
    }
    return 0;
}
