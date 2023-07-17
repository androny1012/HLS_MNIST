#include "conv_net.h"
#include <stdint.h>


#define aver(a,b) \
({ __typeof__ (a) _a = (a); \
   __typeof__ (b) _b = (b); \
 (_a + _b) >> 1; })

uint16_t averFour(uint16_t a, uint16_t b, uint16_t c, uint16_t d) {
	 return aver(aver(a,b), aver(c,d));
}

void pooling_1(uint16_t in[112][112][3], uint16_t out[56][56][3]) {

    uint16_t i, j, m;

    for (m = 0; m < 3; ++m) {
        for (i = 0; i < 56; i++) {
            for (j = 0; j < 56; j++) {
                #pragma HLS UNROLL
            	out[i][j][m] = averFour(
            	        in[i << 1][j << 1][m],
            	        in[(i << 1) + 1][j << 1][m],
            	        in[i << 1][(j << 1) + 1][m],
            	        in[(i << 1) + 1][(j << 1) + 1][m]

                );
            }
        }
    }
}

void pooling_2(uint16_t in[56][56][3], DTYPE out[28][28][3]) {

    uint16_t i, j, m;

    for (m = 0; m < 3; ++m) {
        for (i = 0; i < 28; i++) {
            for (j = 0; j < 28; j++) {
                #pragma HLS UNROLL
            	out[i][j][m] = averFour(
            	        in[i << 1][j << 1][m],
            	        in[(i << 1) + 1][j << 1][m],
            	        in[i << 1][(j << 1) + 1][m],
            	        in[(i << 1) + 1][(j << 1) + 1][m]

                );
            }
        }
    }
}

