#include "conv_net.h"
#include <stdint.h>

#define max(a,b) \
({ __typeof__ (a) _a = (a); \
   __typeof__ (b) _b = (b); \
 _a > _b ? _a : _b; })

DTYPE maxFour(DTYPE a, DTYPE b, DTYPE c, DTYPE d) {
    return max(max(a,b), max(c,d));
}

void pool1_layer(
	DTYPE in[C1_OUT_DMNIN][C1_OUT_DMNIN][C1_N_FILTERS],
	DTYPE out[C2_X_DMNIN][C2_X_DMNIN][C1_N_FILTERS])
{

    uint8_t i, j, m;

    Channel:
    for (m = 0; m < C1_N_FILTERS; m++) {
    	Row:
        for (i = 0; i < C2_X_DMNIN; i++) {
        	Col:
            for (j = 0; j < C2_X_DMNIN; j++) {
                #pragma HLS UNROLL
            	out[i][j][m] = maxFour(
            	        in[i << 1][j << 1][m],
            	        in[(i << 1) + 1][j << 1][m],
            	        in[i << 1][(j << 1) + 1][m],
            	        in[(i << 1) + 1][(j << 1) + 1][m]

                );
            }
        }
    }
}

void pool2_layer(
		DTYPE in[C2_OUT_DMNIN][C2_OUT_DMNIN][C2_N_FILTERS],
		DTYPE out[C3_X_DMNIN][C3_X_DMNIN][C2_N_FILTERS])
{

    uint8_t i, j, m;

    Channel:
    for (m = 0; m < C2_N_FILTERS; m++) {
    	Row:
        for (i = 0; i < C3_X_DMNIN; i++) {
        	Col:
            for (j = 0; j < C3_X_DMNIN; j++) {
                #pragma HLS UNROLL
                out[i][j][m] = maxFour(
                        in[i << 1][j << 1][m],
                        in[(i << 1) + 1][j << 1][m],
                        in[i << 1][(j << 1) + 1][m],
                        in[(i << 1) + 1][(j << 1) + 1][m]
                );
            }
        }
    }
}


