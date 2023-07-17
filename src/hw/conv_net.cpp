#include "conv_net.h"
#include "weights.h"
#include "biases.h"
#include <stdio.h>
#include <string.h>


void CNN(uint8_t *in_b, uint8_t *in_g, uint8_t *in_r,DTYPE *out_t) {
#pragma HLS INTERFACE s_axilite register port=in_b offset=0X010 bundle=ctl
#pragma HLS INTERFACE m_axi depth=512 port=in_b offset=slave bundle=ioData
#pragma HLS INTERFACE s_axilite register port=in_g offset=0X020 bundle=ctl
#pragma HLS INTERFACE m_axi depth=512 port=in_g offset=slave bundle=ioData
#pragma HLS INTERFACE s_axilite register port=in_r offset=0X030 bundle=ctl
#pragma HLS INTERFACE m_axi depth=512 port=in_r offset=slave bundle=ioData
#pragma HLS INTERFACE s_axilite register port=out_t offset=0X040 bundle=ctl
#pragma HLS INTERFACE m_axi depth=512 port=out_t offset=slave bundle=ioData
#pragma HLS INTERFACE s_axilite register port=return  bundle=ctl

	uint16_t w,h,ci,co;
	DTYPE img[IMG_DMNIN][IMG_DMNIN][IMG_CHANNELS];
	uint8_t in_buffer[IMG_DMNIN*IMG_DMNIN];
	DTYPE p[1];

	memcpy(in_buffer,in_r,IMG_DMNIN*IMG_DMNIN*sizeof(uint8_t));
	for (h = 0; h < IMG_DMNIN; h++)
		for (w = 0; w < IMG_DMNIN; w++){
			img[h][w][0] = in_buffer[h*IMG_DMNIN+w];
		}

	predict(img,p);
    out_t[0]=p[0];

}

void predict(DTYPE img[IMG_DMNIN][IMG_DMNIN][IMG_CHANNELS],DTYPE p[1]) {

    DTYPE conv1_out[C1_OUT_DMNIN][C1_OUT_DMNIN][C1_N_FILTERS];
    DTYPE relu1_out[C1_OUT_DMNIN][C1_OUT_DMNIN][C1_N_FILTERS];
    DTYPE pool1_out[C2_X_DMNIN][C2_X_DMNIN][C1_N_FILTERS];

    DTYPE conv2_out[C2_OUT_DMNIN][C2_OUT_DMNIN][C2_N_FILTERS];
    DTYPE relu2_out[C2_OUT_DMNIN][C2_OUT_DMNIN][C2_N_FILTERS];
    DTYPE pool2_out[C3_X_DMNIN][C3_X_DMNIN][C2_N_FILTERS];

    DTYPE pool2_out_flatten[F1_COLS];
    DTYPE fc1_out[F1_ROWS];

    conv1_layer(img, conv1_weight, conv1_out, biases_C1);
    relu1_layer(conv1_out, relu1_out);
    pool1_layer(relu1_out, pool1_out);

    conv2_layer(pool1_out, conv2_weight, conv2_out, biases_C2);
    relu2_layer(conv2_out, relu2_out);
    pool2_layer(relu2_out, pool2_out);

    flatten(pool2_out, pool2_out_flatten);
    fc1_layer(pool2_out_flatten, weights_F1, biases_F1, fc1_out);
    fc1_max(fc1_out,p);
//    for (int i = 0; i < F1_ROWS; i++)
//    	printf("%d\n",fc1_out[i]);

}
