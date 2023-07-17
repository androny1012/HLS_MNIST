#include "conv_net.h"
#include <stdint.h>

void relu1_layer(DTYPE in[C1_OUT_DMNIN][C1_OUT_DMNIN][C1_N_FILTERS], DTYPE out[C1_OUT_DMNIN][C1_OUT_DMNIN][C1_N_FILTERS]) {

    uint16_t r;
    uint8_t c, m;

    for (r = 0; r < C1_OUT_DMNIN; ++r) {
    	for (c = 0; c < C1_OUT_DMNIN; ++c) {
    		relu_a1:for (m = 0; m < C1_N_FILTERS; ++m) {
    			out[r][c][m] = (in[r][c][m] > 0) ? ( in[r][c][m]>>8 ) : 0;
            }
        }
    }
}

void relu2_layer(DTYPE in[C2_OUT_DMNIN][C2_OUT_DMNIN][C2_N_FILTERS], DTYPE out[C2_OUT_DMNIN][C2_OUT_DMNIN][C2_N_FILTERS]) {

    uint16_t r;
    uint8_t c, m;

    for (r = 0; r < C2_OUT_DMNIN; ++r) {
    	for (c = 0; c < C2_OUT_DMNIN; ++c) {
    		relu_a2:for (m = 0; m < C2_N_FILTERS; ++m) {
                out[r][c][m] = (in[r][c][m] > 0) ? (in[r][c][m]>>8) : 0;
            }
        }
    }
}
