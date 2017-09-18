// The maximum number of threads in a block
const int WARP_SIZE = 32;
const int MAX_BLOCK_SIZE = 512;

// Number of threads in a block given an input size up to MAX_BLOCK_SIZE
static int getNumThreads(int nElem) {
  int threadSizes[5] = { 32, 64, 128, 256, MAX_BLOCK_SIZE };
  for (int i = 0; i != 5; ++i) {
    if (nElem <= threadSizes[i]) {
      return threadSizes[i];
    }
  }
  return MAX_BLOCK_SIZE;
}

__device__ __forceinline__ int getMSB(int val) {
  return 31 - __clz(val);
}
