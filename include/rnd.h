#ifndef RND_H
#define RND_H
#include <bits/stdc++.h>
std::mt19937_64 mr(std::chrono::system_clock::now().time_since_epoch().count());
/* mt19937_64 mr(65536); */
double rd(double l, double r) { return std::uniform_real_distribution<double>(l, r)(mr); }
double nd(double l, double r) { return std::normal_distribution<double>(l, r)(mr); }
int ri(int l, int r) { return std::uniform_int_distribution<int>(l, r)(mr); }
#endif
