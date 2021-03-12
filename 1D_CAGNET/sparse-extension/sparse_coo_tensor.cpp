#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Layout.h>
#include <ATen/Parallel.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/NativeFunctions.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/SparseTensorUtils.h>

#include "cusparse.h"

#include <pybind11/pybind11.h>

#include <THC/THCGeneral.hpp>

#include <torch/extension.h>

namespace py = pybind11;

using namespace at::sparse;

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
    }                                                                          \
}

#define CHECK_ERROR(str) \
    {cudaDeviceSynchronize(); cudaError_t err; err = cudaGetLastError(); if(err!=0) {printf("ERROR %s:  %d %s\n", str, err, cudaGetErrorString(err)); fflush(stdout);}}


at::Tensor expand_values_if_needed(const at::Tensor& values) {
    // expand
    if (values.dim() == 0) {
        // Mimic Numpy behavior here and treat it as a 1D tensor
        return values.expand({1});
    } else {
        return values;
    }
}

at::Tensor sparse_coo_tensor_gpu(const at::Tensor& indices, 
                                    const at::Tensor& values_, 
                                    at::ArrayRef<int64_t> size) {

    at::Tensor values = expand_values_if_needed(values_); 

    int64_t sparse_dim = indices.size(0);
    int64_t dense_dim = values.dim() - 1;

    return at::_sparse_coo_tensor_with_dims_and_tensors(
        sparse_dim, dense_dim, size, indices, values, values.options().layout(at::kSparse));
}

template<typename T>
void printCusparseDnMat(int64_t rows, int64_t cols, int64_t ld, T *values_dev) {
  T* values_host = new T[rows*cols];
  cudaMemcpy(values_host, values_dev, rows*cols*sizeof(T), cudaMemcpyDeviceToHost);
  for (int64_t row = 0; row < rows; row++) {
    for (int64_t col = 0; col < cols; col++) {
      // Cusparse dense matrices are stored in column-major order
      std::cout << values_host[col*rows+row] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "  values: ";
  for (int64_t i = 0; i < rows*cols; i++) {
    std::cout << values_host[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "  shape: " << rows << ", " << cols << std::endl;
  delete [] values_host;
}

template<typename T>
void printCusparseSpMat(int32_t rows, int32_t cols, int32_t nnz, int32_t *row_indices_dev,
                            int32_t *col_indices_dev, T *values_dev) {
  T* values_host = new T[nnz];
  int32_t* row_indices_host = new int32_t[nnz];
  int32_t* col_indices_host = new int32_t[nnz];
  cudaMemcpy(values_host, values_dev, nnz*sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(row_indices_host, row_indices_dev, nnz*sizeof(int32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(col_indices_host, col_indices_dev, nnz*sizeof(int32_t), cudaMemcpyDeviceToHost);

  for (int64_t i = 0; i < nnz; i++) {
    std::cout << "(" << row_indices_host[i]
      << ", " << col_indices_host[i]
      << "): " << values_host[i] << std::endl;
  }
  std::cout << "  values: ";
  for (int64_t i = 0; i < nnz; i++) {
    std::cout << values_host[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "  row_indices: ";
  for (int64_t i = 0; i < nnz; i++) {
    std::cout << row_indices_host[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "  col_indices: ";
  for (int64_t i = 0; i < nnz; i++) {
    std::cout << col_indices_host[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "  shape: " << rows << ", " << cols << std::endl;
  delete [] values_host;
  delete [] row_indices_host;
  delete [] col_indices_host;
}

// at::Tensor spmm_gpu(const at::Tensor& A_rowindices, 
void spmm_gpu(const at::Tensor& A_rowindices, 
                        const at::Tensor& A_colindices,
                        const at::Tensor& A_values, 
                        int32_t n,
                        int32_t m,
                        at::Tensor& B,
                        at::Tensor& C) {
    // return;

    // cusparseHandle_t handle;
    // CHECK_CUSPARSE(cusparseCreate(&handle));
    auto state = at::globalContext().lazyInitCUDA();
    // auto handle = THCState_getCurrentSparseHandle(state);
    auto handle = at::cuda::getCurrentCUDASparseHandle();

    // Impl1 -- coo2csr + csrmm2
    int nnz = A_values.size(0);

    clock_t start, stop;
    
    int32_t *d_a_csrrows;

    //  cudaMalloc(&d_a_csrrows, (n + 1) * sizeof(int32_t));
    // CHECK_CUSPARSE(cusparseXcoo2csr(handle, 
                                        //A_rowindices.data_ptr<int>(), 
                                        //nnz, 
                                        //n, 
                                        //d_a_csrrows, 
                                        //CUSPARSE_INDEX_BASE_ZERO));


    float *dA_values, *dB, *dC;
    dB = B.data_ptr<float>();
    float alpha = 1;
    float beta = 1;
    // cusparseMatDescr_t descrA;
    cusparseDnMatDescr_t matB, matC;
    void* dBuffer    = NULL;
    size_t bufferSize = 0;
    // cusparseCreateMatDescr(&descrA);
    // cusparseSetMatType(descrA,CUSPARSE_MATRIX_TYPE_GENERAL);

    // cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO);

    cusparseSpMatDescr_t matA;
    CHECK_CUSPARSE( cusparseCreateCoo(&matA, n, m, nnz,
                                      A_rowindices.data_ptr<int>(), A_rowindices.data_ptr<int>(),
                                      A_values.data_ptr<float>(),
                                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )

    // auto C = torch::ones({n, B.size(1)}, torch::dtype(torch::kDouble).device(torch::kCUDA));
    //

    // n A.size(0), row
    // m A.size(1), col
    int32_t b_row = B.size(0);
    int32_t b_col = B.size(1);
    int32_t c_row = C.size(0);
    int32_t c_col = C.size(1);

    assert(m==b_row);
    assert(n==c_row);
    assert(b_col==c_col);
    assert(m==b_row);

    //
    // A n* b_col
    // B  b_col * m  BT m* b_col
    // C b_col * m
    

    int   ldb             = b_col;
    // B from torch is row major
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, b_col,  b_row, ldb, dB,
        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    // Row-major to column-major
    // B.t_();
    // B.set_data(B.contiguous());
    // B.set_data(B.view({b_row, b_col}));
    // C.contiguous().t_();
    // C.set_data(C.contiguous().t_().view({c_row, c_col}));
    C.set_data(C.t_().contiguous());
    //C.set_data(C.view({c_row, c_col}));

    dC = C.data_ptr<float>();
    int   ldc             = c_row;  // should be A row

    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, c_row, c_col, ldc, dC,
        CUDA_R_32F, CUSPARSE_ORDER_COL) )


    CHECK_CUSPARSE(cusparseSpMM(handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_TRANSPOSE,
                                    &alpha,
                                    matA,
                                    matB,
                                    &beta,
                                    matC,
                                    CUDA_R_32F,
                                    CUSPARSE_SPMM_ALG_DEFAULT,
                                    dBuffer)); 

    /*
      CHECK_CUSPARSE(cusparseScsrmm2(handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_TRANSPOSE,
                                    n,
                                    b_col,
                                    m,
                                    nnz,
                                    &alpha,
                                    descrA,
                                    A_values.data_ptr<float>(),
                                    d_a_csrrows,
                                    A_colindices.data_ptr<int>(),
                                    B.data_ptr<float>(),
                                    B.size(1),
                                    &beta,
                                    C.data_ptr<float>(),
                                    n)); 
                                    */
    // cudaFree(d_a_csrrows);
    // CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    // CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    // CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    // CHECK_CUSPARSE( cusparseDestroy(handle) )

    // Column-major to row-major
    // B.set_data(B.view({b_col, b_row}));
    // B.t_();
    C.set_data(C.t_().contiguous().view({c_row, c_col}));
    // C.t_();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_coo_tensor_gpu", &sparse_coo_tensor_gpu, "Sparse Tensor GPU-only constructor");
    m.def("spmm_gpu", &spmm_gpu, "SpMM wrapper for cusparse");
}
