// #pragma GCC optimize("O3,unroll-loops")
// #pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
// #pragma GCC target("sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2,bmi,bmi2,lzcnt,popcnt")
// #define EIGEN_VECTORIZE_SSE4_2

#ifdef LOCAL
#define dbg(x) cerr << #x << " = " << (x) << endl
#else
#define dbg(...) 42
#define NDEBUG
#endif
#include <bits/stdc++.h>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace std;

using namespace Eigen;
using vec = VectorXf;
using mat = MatrixXf;
using ten1 = Tensor<float, 1>;
using ten2 = Tensor<float, 2>;
using ten3 = Tensor<float, 3>;
using ten4 = Tensor<float, 4>;

ten3 conv(const ten3 &input, const ten4 &kernel) {
    int sz1 = input.dimension(1) - kernel.dimension(2) + 1;
    int sz2 = input.dimension(2) - kernel.dimension(3) + 1;
    ten3 res(kernel.dimension(1), sz1, sz2);
    res.setZero();
    for (int i = 0; i < kernel.dimension(0); i++)
        for (int j = 0; j < kernel.dimension(1); j++)
            res.chip(j, 0) += input.chip(i, 0).convolve(kernel.chip(i, 0).chip(j, 0), std::array<int, 2>{0, 1});
    return res;
}

int main() {
    // ten3 x(2, 2, 2), y(1, 1, 1);
    // ten4 core(2, 1, 2, 2);
    // x.setValues({{{1, 1}, {1, 1}}, {{1, 1}, {1, 1}}});
    // core.setValues({{{{1, 1}, {1, 1}}}, {{{1, 1}, {1, 1}}}});
    // cout << core.chip(0, 0);
    // y = conv(x, core);
    // cout << y;

    ten3 x(1, 2, 3);
    Tensor<float, 0> tmp = (x.chip(0, 0).sum());
    float a=Tensor<float, 0>(x.chip(0, 0).sum())();
    // mat x(3, 3); x.resize(3,4);
    // TensorMap<ten3> tmp(x.data(), {4, 1, 3});
    return 0;
}
