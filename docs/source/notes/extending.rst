Extending PyTorch-Encoding
==========================

In this note we'll discuss extending PyTorch-Encoding package,
which is extending :mod:`torch.nn` and
:mod:`torch.autograd` with custom CUDA backend.

Torch C and CUDA Backend
------------------------

Given a simple example of the residual operation (in a mini-batch): 

.. math::
    r_{ik} = x_i - c_k

where the inputs are :math:`X=\{x_1, ...x_N\}` and :math:`C=\{c_1,...c_k\}` and the output is :math:`R=\{r_{ik}\}`. 


- Add CUDA kernel function and expose a C API to the generic file ``encoding/kernel/generic/encoding_kernel.c`` using Torch generic files::

    __global__ void Encoding_(Residual_Forward_kernel) (
        THCDeviceTensor<real, 4> R,
        THCDeviceTensor<real, 3> X,
        THCDeviceTensor<real, 2> D)
    /*
     * residual forward kernel function
     */
    {
        /* declarations of the variables */
        int b, k, d, i, K;
        /* Get the index and channels */ 
        b = blockIdx.z;
        d = blockIdx.x * blockDim.x + threadIdx.x;
        i = blockIdx.y * blockDim.y + threadIdx.y;
        K = R.getSize(2);
        /* boundary check for output */
        if (d >= X.getSize(2) || i >= X.getSize(1))    return;
        /* main operation */
        for(k=0; k<K; k++) {
            R[b][i][k][d] = X[b][i][d].ldg() - D[k][d].ldg();
        }
    }

    void Encoding_(Residual_Forward)(
        THCState *state, THCTensor *R_, THCTensor *X_, THCTensor *D_)
    /*
     * residual forward 
     */
    {
        /* Check the GPU index and tensor dims*/
        THCTensor_(checkGPU)(state, 3, R_, X_, D_); 
        if (THCTensor_(nDimension)(state, R_) != 4 ||
            THCTensor_(nDimension)(state, X_) != 3 ||
            THCTensor_(nDimension)(state, D_) != 2)
        THError("Encoding: incorrect input dims. \n");
        /* Device tensors */
        THCDeviceTensor<real, 4> R = devicetensor<4>(state, R_);
        THCDeviceTensor<real, 3> X = devicetensor<3>(state, X_);
        THCDeviceTensor<real, 2> D = devicetensor<2>(state, D_);
        /* kernel function */
        cudaStream_t stream = THCState_getCurrentStream(state);
        dim3 threads(16, 16);
        dim3 blocks(X.getSize(2)/16+1, X.getSize(1)/16+1, 
                    X.getSize(0));
        Encoding_(Residual_Forward_kernel)<<<blocks, threads, 0, stream>>>(R, X, D);
        THCudaCheck(cudaGetLastError());
    }

- Add corresponding function header to ``encoding/kernel/generic/encoding_kernel.h``::

    void Encoding_(Residual_Forward)(
        THCState *state, THCTensor *R_, THCTensor *X_, THCTensor *D_);

- Add a CFFI function to ``encoding/src/generic/encoding_generic.c``, which calls the C API we just write::

    int Encoding_(residual_forward)(THCTensor *R, THCTensor *X, THCTensor *D)
    /*
     * Residual operation
     */
    {
        Encoding_(Residual_Forward)(state, R, X, D);
        /* C function return number of the outputs */
        return 0;
    }

- Add corresponding function header to ``encoding/src/encoding_lib.h``::
    
    int Encoding_Float_residual_forward(THCudaTensor *R, THCudaTensor *X, 
        THCudaTensor *D);

- Finally, call this function using python::

    class residual(Function):
        def forward(self, X, C):
            # X \in(BxNxD) D \in(KxD) R \in(BxNxKxD) 
            B, N, D = X.size()
            K = C.size(0)
            with torch.cuda.device_of(X):
                R = X.new(B,N,K,D)
            if isinstance(X, torch.cuda.FloatTensor):
                with torch.cuda.device_of(X):
                    encoding_lib.Encoding_Float_residual_forward(R, X, C)
            elif isinstance(X, torch.cuda.DoubleTensor):
                with torch.cuda.device_of(X):
                    encoding_lib.Encoding_Double_residual_forward(R, X, C)
            else:
                raise RuntimeError('Unimplemented data type!')
            return R

- Note this is just an example. You also need to implement backward function for ``residual`` operation. 
