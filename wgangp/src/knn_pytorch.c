#include <THC/THC.h>
#include "knn_cuda_kernel.h"

extern THCState *state;

int knn(THCudaTensor *ref_tensor, THCudaTensor *query_tensor,
    THCudaLongTensor *idx_tensor, THCudaTensor *dist_tensor) {

  THCAssertSameGPU(THCudaTensor_checkGPU(state, 4, idx_tensor, dist_tensor, ref_tensor, query_tensor));
  long ref_nb, query_nb, dim, k;
  THArgCheck(ref_tensor->nDimension == 2 , 0, "ref_tensor: 2D Tensor expected");
  THArgCheck(query_tensor->nDimension == 2 , 1, "query_tensor: 2D Tensor expected");
  THArgCheck(idx_tensor->nDimension == 2 , 3, "idx_tensor: 2D Tensor expected");
  THArgCheck(dist_tensor->nDimension == 2 , 4, "dist_tensor: 2D Tensor expected");
  THArgCheck(ref_tensor->size[0] == query_tensor->size[0], 0, "input sizes must match");
  THArgCheck(idx_tensor->size[0] == dist_tensor->size[0], 0, "output sizes must match");

  ref_tensor = THCudaTensor_newContiguous(state, ref_tensor);
  query_tensor = THCudaTensor_newContiguous(state, query_tensor);

  dim = ref_tensor->size[0];
  k = idx_tensor->size[0];
  ref_nb = ref_tensor->size[1];
  query_nb = query_tensor->size[1];

  float *ref_dev = THCudaTensor_data(state, ref_tensor);
  float *query_dev = THCudaTensor_data(state, query_tensor);
  long *idx_dev = THCudaLongTensor_data(state, idx_tensor);
  float *dist_dev = THCudaTensor_data(state, dist_tensor);

  knn_device(ref_dev, ref_nb, query_dev, query_nb, dim, k, dist_dev, idx_dev,
    THCState_getCurrentStream(state));

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in knn: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }

  return 1;
}
