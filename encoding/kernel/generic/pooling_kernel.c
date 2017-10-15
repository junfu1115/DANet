/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Created by: Hang Zhang
 * ECE Department, Rutgers University
 * Email: zhang.hang@rutgers.edu
 * Copyright (c) 2017
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree 
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 */
#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/pooling_kernel.c"
#else


__global__ void Encoding_(DilatedAvgPool_Forward_kernel) (
    THCDeviceTensor<real, 4> X, 
    THCDeviceTensor<real, 4> Y, 
    int kH, int kW, int dH, int dW,
    int padH, int padW, int dilationH, int dilationW
    )
/*
 * dilated avgpool2d forward kernel function
 */
{
    /* declarations of the variables */
    int bc, b, c, w, h, C;
    real sum;
    /* Get the index and channels */ 
    bc = blockIdx.z;
    w = blockIdx.x * blockDim.x + threadIdx.x;
    h = blockIdx.y * blockDim.y + threadIdx.y;
    C = Y.getSize(1);
    b = bc / C;
    c = bc - b*C;
    /* boundary check for output */
    if (w >= Y.getSize(3) || h >= Y.getSize(2)) return;
    int hstart = h*dW -padH;
    int wstart = w*dW -padW;
    int hend = min(hstart + kH*dilationH, X.getSize(2));
    int wend = min(wstart + kW*dilationW, X.getSize(3));
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    int pool_size = ((hend - hstart - 1) / dilationH + 1) * 
        ((wend - wstart - 1) / dilationW + 1);
    sum = 0;
    for (int th=hstart; th < hend; th+=dilationH) {
        for (int tw=wstart; tw < wend; tw+=dilationW) {
            sum += X[b][c][th][tw];
        }
    }
    Y[b][c][h][w] = sum / pool_size;
}

void Encoding_(DilatedAvgPool_Forward)(THCState *state, 
    THCTensor *X_, THCTensor *Y_, 
    int kH, int kW, int dH, int dW,
    int padH, int padW,
    int dilationH, int dilationW)
/*
 * dilated avgpool2d forward function
 */
{
    /* Check the GPU index and tensor dims*/
    THCTensor_(checkGPU)(state, 2, X_, Y_);
    if (THCTensor_(nDimension)(state, X_) != 4 ||
        THCTensor_(nDimension)(state, Y_) != 4)
        THError("Encoding: incorrect input dims. \n");
    /* Device tensors */
    THCDeviceTensor<real, 4> X = devicetensor<4>(state, X_);
    THCDeviceTensor<real, 4> Y = devicetensor<4>(state, Y_);
    /* kernel function */
    cudaStream_t stream = THCState_getCurrentStream(state);
    dim3 threads(16, 16);
    dim3 blocks(Y.getSize(3)/16+1, Y.getSize(2)/16+1, 
                Y.getSize(1)*Y.getSize(0));
    Encoding_(DilatedAvgPool_Forward_kernel)<<<blocks, threads, 0, stream>>>
        (X, Y, kH, kW, dH, dW, padH, padW, dilationH, dilationW);
    THCudaCheck(cudaGetLastError());
}

__global__ void Encoding_(DilatedAvgPool_Backward_kernel) (
    THCDeviceTensor<real, 4> gradX, 
    THCDeviceTensor<real, 4> gradY, 
    int kH, int kW, int dH, int dW,
    int padH, int padW, int dilationH, int dilationW
    )
/*
 * dilated avgpool2d forward kernel function
 */
{
    /* declarations of the variables */
    int bc, b, c, w, h, C;
    real sum;
    /* Get the index and channels */ 
    bc = blockIdx.z;
    w = blockIdx.x * blockDim.x + threadIdx.x;
    h = blockIdx.y * blockDim.y + threadIdx.y;
    C = gradX.getSize(1);
    b = bc / C;
    c = bc - b*C;
    /* boundary check for output */
    if (w >= gradX.getSize(3) || h >= gradX.getSize(2)) return;
    int phstart = (h + padH < ((kH-1)*dilationH+1)) ? 0 : 
        (h + padH - ((kH-1)*dilationH+1))/dH + 1;
    int pwstart = (w + padW < ((kW-1)*dilationW+1)) ? 0 : 
        (w + padW - ((kW-1)*dilationW+1))/dW + 1;
    int phend = min((h+padH)/dH+1, gradY.getSize(2));
    int pwend = min((w+padW)/dW+1, gradY.getSize(3));
    sum = 0;
    int hstart, wstart, hend, wend, pool_size;
    for (int ph=phstart; ph < phend; ++ph) {
        for (int pw=pwstart; pw < pwend; ++pw) {
            hstart = ph*dW -padH;
            wstart = pw*dW -padW;
            hend = min(hstart + kH*dilationH, gradX.getSize(2));
            wend = min(wstart + kW*dilationW, gradX.getSize(3));
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            pool_size = ((hend - hstart - 1) / dilationH + 1) * 
                ((wend - wstart - 1) / dilationW + 1);
            sum += gradY[b][c][ph][pw] / pool_size;
        }
    }
    gradX[b][c][h][w] = sum;
}

void Encoding_(DilatedAvgPool_Backward)(THCState *state, 
    THCTensor *gradX_, THCTensor *gradY_, 
    int kH, int kW, int dH, int dW,
    int padH, int padW,
    int dilationH, int dilationW)
/*
 * dilated avgpool2d forward function
 */
{
    /* Check the GPU index and tensor dims*/
    THCTensor_(checkGPU)(state, 2, gradX_, gradY_);
    if (THCTensor_(nDimension)(state, gradX_) != 4 ||
        THCTensor_(nDimension)(state, gradY_) != 4)
        THError("Encoding: incorrect input dims. \n");
    /* Device tensors */
    THCDeviceTensor<real, 4> gradX = devicetensor<4>(state, gradX_);
    THCDeviceTensor<real, 4> gradY = devicetensor<4>(state, gradY_);
    /* kernel function */
    cudaStream_t stream = THCState_getCurrentStream(state);
    dim3 threads(16, 16);
    dim3 blocks(gradX.getSize(3)/16+1, gradX.getSize(2)/16+1, 
                gradX.getSize(1)*gradX.getSize(0));
    Encoding_(DilatedAvgPool_Backward_kernel)<<<blocks, threads, 0, stream>>>
        (gradX, gradY, kH, kW, dH, dW, padH, padW, dilationH, dilationW);
    THCudaCheck(cudaGetLastError());
}

#endif
