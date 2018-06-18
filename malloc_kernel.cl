
/* kernel.cl 
 * Matrix multiplication: C = A * B.
 * Device code.
 */
 
// OpenCL Kernel
__kernel void
matrixMul(__global float* C,
          __global float* A, 
          __global float* B, 
          __global float* D, 
          __global float* E, 
          int wA, int wB, int wC, int wD, int WE)
{
  
   int tx = get_global_id(0); 
   int ty = get_global_id(1);
 
   // value stores the element that is 
   // computed by the thread
   float value = 0;
   for (int k = 0; k < wA; ++k)
   {
      float elementA = A[ty * wA + k];
      float elementB = B[k * wB + tx];
      value += elementA * elementB;
   }
      
   // Write the matrix to device memory each thread writes one element & Sigmoid
   float val = 0;
   val = 1/(1+exp(-value));
   C[ty * wA + tx] = val;

   for (int k = 0; k < wC; ++k)
   {
      float elementC = C[ty * wC + k];
      float elementD = D[k * wD + tx];
      value += elementC * elementD;
   }

   val = 1/(1+exp(-value));
   E[ty * wA + tx] = val;
}
