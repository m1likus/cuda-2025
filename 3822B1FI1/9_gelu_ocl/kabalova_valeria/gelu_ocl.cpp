#include "gelu_ocl.h"
#include <CL/cl.h>
#include <cstring>

const char* source = R"(
  __kernel void geluKernel(__global const float* input, __global float* result, const int n) { 
    int id = get_global_id(0);
    if (id < n) {
      float x = input[id];
      result[id] = 0.5f * x * (1.0f + tanh(0.797884f * (x + 0.044715f * (x * x * x))));
    }
  }
)";

std::vector<float> GeluOCL(const std::vector<float>& input) {
  const size_t size = input.size();
  std::vector<float> result(size);
  
  cl_platform_id platform;
  clGetPlatformIDs(1, &platform, nullptr);

  cl_device_id device;
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

  cl_context context;
  context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);

  cl_command_queue queue;
  queue = clCreateCommandQueue(context, device, 0, nullptr);

  cl_mem in, out;
  in = clCreateBuffer(context, CL_MEM_READ_ONLY, size * sizeof(float), nullptr, nullptr);
  out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size * sizeof(float), nullptr, nullptr);
  clEnqueueWriteBuffer(queue, in, CL_TRUE, 0, size * sizeof(float), input.data(), 0, nullptr, nullptr);

  cl_program program = clCreateProgramWithSource(context, 1, &source, nullptr, nullptr);
  clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
  cl_kernel clkernel = clCreateKernel(program, "geluKernel", nullptr);

  clSetKernelArg(clkernel, 0, sizeof(cl_mem), &in);
  clSetKernelArg(clkernel, 1, sizeof(cl_mem), &out);
  clSetKernelArg(clkernel, 2, sizeof(int), &size);

  size_t localSize = 256;
  size_t globalSize = (size + localSize - 1) / localSize * localSize;
  clEnqueueNDRangeKernel(queue, clkernel, 1, nullptr, &globalSize, &localSize, 0, nullptr, nullptr);
  clFinish(queue);

  clEnqueueReadBuffer(queue, out, CL_TRUE, 0, size * sizeof(float), result.data(), 0, nullptr, nullptr);

  clReleaseMemObject(in);
  clReleaseMemObject(out);
  clReleaseProgram(program);
  clReleaseKernel(clkernel);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  return result;
}