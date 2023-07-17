#include "conv_net.h"
#include <stdint.h>

void fc1_max(DTYPE Z[F1_ROWS], DTYPE p[1]) {

    uint8_t i,k=0;
    int32_t idx[F1_ROWS];
    DTYPE max=Z[0];

    for (i = 0; i < F1_ROWS; ++i) {
    	if(Z[i]>max && Z[i] > 0) { k = i; max=Z[i];}
    }

    p[0] = k;
}
