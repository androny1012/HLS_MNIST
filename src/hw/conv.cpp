#include "conv_net.h"
#include <stdint.h>

void conv1_layer (
		      DTYPE   X[C1_X_DMNIN][C1_X_DMNIN][C1_N_CHAN],
        const DTYPE   W[C1_W_DMNIN][C1_W_DMNIN][C1_N_CHAN][C1_N_FILTERS],
              DTYPE out[C1_OUT_DMNIN][C1_OUT_DMNIN][C1_N_FILTERS],
        const DTYPE bias[C1_N_FILTERS]) {

    uint8_t ch, f, i, j, r, c;

    for (f = 0; f < C1_N_FILTERS; ++f) {
    	for (r = 0; r < C1_OUT_DMNIN; ++r) {
    		for (c = 0; c < C1_OUT_DMNIN; ++c) {
				#pragma HLS PIPELINE
    				out[r][c][f] = bias[f];
    		}
    	}
	}
//    Output_Channel:
//	for (f = 0; f < C1_N_FILTERS; ++f) {
//		Input_Channel:
//		for (ch = 0; ch < C1_N_CHAN; ++ch) {
//			Row:
//			for (r = 0; r < C1_X_DMNIN - C1_W_DMNIN + 1; r += STRIDE) {
//				Column:
//				for (c = 0, i = 0, j = 0; c < C1_X_DMNIN - C1_W_DMNIN + 1; c += STRIDE) {
//					Kernel_Row:
//					for (i = 0; i < C1_W_DMNIN; ++i) {
//						Kernel_Column:
//						for (j = 0; j < C1_W_DMNIN; ++j) {
//                        	out[r][c][f] += W[i][j][ch][f] * X[r + i][j + c][ch];
//                        }
//                    }
//                }
//            }
//        }
//    }


    			Row:
    			for (r = 0; r < C1_X_DMNIN - C1_W_DMNIN + 1; r += STRIDE) {
    				Column:
    				for (c = 0, i = 0, j = 0; c < C1_X_DMNIN - C1_W_DMNIN + 1; c += STRIDE) {
    					Kernel_Row:
    					for (i = 0; i < C1_W_DMNIN; ++i) {
    						Kernel_Column:
    						for (j = 0; j < C1_W_DMNIN; ++j) {
    					        Output_Channel:
    					    	for (f = 0; f < C1_N_FILTERS; ++f) {
#pragma HLS UNROLL
    					    		Input_Channel:
    					    		for (ch = 0; ch < C1_N_CHAN; ++ch) {
#pragma HLS UNROLL
                            	out[r][c][f] += W[i][j][ch][f] * X[r + i][j + c][ch];
                            }
                        }
                    }
                }
            }
        }
}


void conv2_layer (
              DTYPE   X[C2_X_DMNIN][C2_X_DMNIN][C2_N_CHAN],
        const DTYPE   W[C2_W_DMNIN][C2_W_DMNIN][C2_N_CHAN][C2_N_FILTERS],
              DTYPE out[C2_OUT_DMNIN][C2_OUT_DMNIN][C2_N_FILTERS],
        const DTYPE bias[C2_N_FILTERS]) {

    uint8_t ch, f, i, j, r, c;

    for (f = 0; f < C2_N_FILTERS; ++f) {
        for (r = 0; r < C2_OUT_DMNIN; ++r) {
            for (c = 0; c < C2_OUT_DMNIN; ++c) {
                #pragma HLS PIPELINE
                out[r][c][f] = bias[f];
            }
        }
    }

        	Row:
            for (r = 0; r < C2_X_DMNIN - C2_W_DMNIN + 1; r += STRIDE) {
            	Column:
                for (c = 0, i = 0, j = 0; c < C2_X_DMNIN - C2_W_DMNIN + 1; c += STRIDE) {
                	Kernel_Row:
                    for (i = 0; i < C2_W_DMNIN; ++i) {
                    	Kernel_Column:
                        for (j = 0; j < C2_W_DMNIN; ++j) {
                        	Output_Channel:
                            for (f = 0; f < C2_N_FILTERS; ++f) {
#pragma HLS UNROLL
                            	Input_Channel:
                                for (ch = 0; ch < C2_N_CHAN; ++ch) {
#pragma HLS UNROLL
                                	out[r][c][f] += W[i][j][ch][f] * X[r + i][j + c][ch];

}}}}}}
}
