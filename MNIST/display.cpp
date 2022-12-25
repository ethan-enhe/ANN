#include <bits/stdc++.h>

#include "../include/microdnn.h"
#include "./BMPImage.h"

using namespace std;

sequential net;
int main() {
    net.add(make_shared<linear>(28 * 28, 300));
    net.add(make_shared<batchnorm>(300));

    net.add(make_shared<relu>());
    net.add(make_shared<linear>(300, 128));

    net.add(make_shared<relu>());
    net.add(make_shared<batchnorm>(128));
    net.add(make_shared<linear>(128, 64));

    net.add(make_shared<relu>());
    net.add(make_shared<linear>(64, 10));

    net.add(make_shared<softmax>());

    net.readf("model_acc98.71.txt");
    net.set_train_mode(0);

    while (1) {
        puts("press enter after drawing");
        getchar();
        puts("recognizing...");

        vec input(28 * 28), res;
        BMPImage bmpImage1("./input.bmp");
        for (uint4 i = 0; i < bmpImage1.height(); i++) {
            for (uint4 j = 0; j < bmpImage1.width(); j++) {
                BGR value = bmpImage1.at<BGR>(i, j);
                input[i * 28 + j] = value.b;
            }
        }
        res = net.forward(vec_batch{input})[0];
        int mxi = 0;
        for (int i = 0; i < 10; i++)
            if (res[i] > res[mxi]) mxi = i;
        cout << mxi << " " << res[mxi] << endl;
    }

    return 0;
}
