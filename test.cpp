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

int main() {
    // Compute convolution along the second and third dimension.
    ten4 input(3, 3, 7, 11);
    mat kernel(2, 2);
    ten4 output(3, 2, 6, 11);
    ten2 x(2,2);
    x.setRandom();
    cout<<x<<endl;
    cout<<(x+x.constant(1))<<endl;
    // input.setRandom();
    // kernel.setRandom();
    //
    // auto k = Map<mat>(kernel.data(), {2, 2});
    // output = input.convolve(k, std::array<int, 2>{1, 2});

    return 0;
}
