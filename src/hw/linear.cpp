#include "conv_net.h"
#include <stdint.h>

void flatten(
        DTYPE IN[C3_X_DMNIN][C3_X_DMNIN][C2_N_FILTERS],
        DTYPE OUT[F1_COLS]) {

    uint16_t i,j,k;
    uint16_t t = 0;

    for (i = 0; i < C2_N_FILTERS; ++i) {
    	for (j = 0; j < C3_X_DMNIN; ++j) {
    	    flatten:for (k = 0; k < C3_X_DMNIN; ++k) {
                OUT[t++] = IN[j][k][i];
            }
        }
    }
}

void fc1_layer(
          DTYPE X[F1_COLS],
    const DTYPE W[F1_ROWS][F1_COLS],
    const DTYPE bias[F1_ROWS],
          DTYPE Z[F1_ROWS]) {

    uint16_t c;
    uint16_t r;

    for (r = 0; r < F1_ROWS; ++r) {
        Z[r] = bias[r];
        for (c = 0; c < F1_COLS; ++c) {
            Z[r] += X[c] * W[r][c];
        }
    }
}
