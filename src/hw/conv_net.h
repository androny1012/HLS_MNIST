#ifndef CONV_NET_H
#define CONV_NET_H

#include <stdint.h>

typedef int64_t DTYPE;

const uint8_t IMG_DMNIN = 28;
const uint8_t IMG_CHANNELS = 1;
const uint8_t STRIDE = 1;

const uint8_t C1_N_CHAN = 1;
const uint8_t C1_X_DMNIN = 28;
const uint8_t C1_W_DMNIN = 3;
const uint8_t C1_OUT_DMNIN = 26;
const uint8_t C1_N_FILTERS = 8;

const uint8_t C2_N_CHAN = 8;
const uint8_t C2_X_DMNIN = 13;
const uint8_t C2_W_DMNIN = 3;
const uint8_t C2_OUT_DMNIN = 11;
const uint8_t C2_N_FILTERS = 8;

const uint8_t C3_X_DMNIN = 5;
const uint16_t F1_ROWS = 10;
const uint16_t F1_COLS = 200;

void CNN(uint8_t *in_b, uint8_t *in_g, uint8_t *in_r,
					  DTYPE out_t[1]);

void predict(DTYPE img[IMG_DMNIN][IMG_DMNIN][IMG_CHANNELS], DTYPE p[1]);

void conv1_layer (
		      DTYPE    X[C1_X_DMNIN][C1_X_DMNIN][C1_N_CHAN],
        const DTYPE    W[C1_W_DMNIN][C1_W_DMNIN][C1_N_CHAN][C1_N_FILTERS],
              DTYPE  out[C1_OUT_DMNIN][C1_OUT_DMNIN][C1_N_FILTERS],
        const DTYPE bias[C1_N_FILTERS]);


void relu1_layer(
        DTYPE in[C1_OUT_DMNIN][C1_OUT_DMNIN][C1_N_FILTERS],
        DTYPE out[C1_OUT_DMNIN][C1_OUT_DMNIN][C1_N_FILTERS]);

void pool1_layer(
		DTYPE in[C1_OUT_DMNIN][C1_OUT_DMNIN][C1_N_FILTERS],
        DTYPE out[C2_X_DMNIN][C2_X_DMNIN][C1_N_FILTERS]);

void conv2_layer (
              DTYPE    X[C2_X_DMNIN][C2_X_DMNIN][C2_N_CHAN],
        const DTYPE    W[C2_W_DMNIN][C2_W_DMNIN][C2_N_CHAN][C2_N_FILTERS],
              DTYPE  out[C2_OUT_DMNIN][C2_OUT_DMNIN][C2_N_FILTERS],
        const DTYPE bias[C2_N_FILTERS]);

void relu2_layer(
        DTYPE in[C2_OUT_DMNIN][C2_OUT_DMNIN][C2_N_FILTERS],
        DTYPE out[C2_OUT_DMNIN][C2_OUT_DMNIN][C2_N_FILTERS]);

void pool2_layer(
        DTYPE in[C2_OUT_DMNIN][C2_OUT_DMNIN][C2_N_FILTERS],
        DTYPE out[C3_X_DMNIN][C3_X_DMNIN][C2_N_FILTERS]);

void flatten(
        DTYPE IN[C3_X_DMNIN][C3_X_DMNIN][C2_N_FILTERS],
        DTYPE OUT[F1_COLS]);

void fc1_layer(
              DTYPE X[F1_COLS],
        const DTYPE W[F1_ROWS][F1_COLS],
        const DTYPE bias[F1_ROWS],
              DTYPE Z[F1_ROWS]);

void fc1_max(DTYPE Z[F1_ROWS], DTYPE p[1]);

#endif
