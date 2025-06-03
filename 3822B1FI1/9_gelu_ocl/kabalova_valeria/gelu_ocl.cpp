#include "gelu_ocl.h"
#include <CL/cl.h>
#include <cstring>

const char* source =
"__kernel void kernel(__global const float* input, __global float* result, const int n) { \n"\
" int id = get_global_id(0);\n"\
" if (id < n) {\n"\
"   float x = input[id];\n"\
"   float arg1 = 0.79788458347320556640625f * (x + 0.044715f * x * x * x);\n"\
"   float tmp;\n"\
"   if (x <= 0.0f) tmp = tanh(arg1);\n"\
"   else if (arg1 >= 4.1f) tmp = 1.0f;\n"\
"   else {\n"\
"     if (x < 0.0001f) tmp = x;\n"\
"     else {\n"\
"     float x_2 = x * x;\n"\
"     tmp = x * (10395.0f + x_2 * (1260.0f + x_2 * 21.0f));\n"\
"     tmp = tmp / (10395.0f + x_2 * (4725.0f + x_2 * (210.0f + x_2)));\n"\
"     }\n"\
"   }\n"\
"  result[i] = 0.5f * x * (1.0f + tmp);\n"\
"  }\n"\
"}\n";

std::vector<float> GeluOCL(const std::vector<float>& input) {
  std::vector<float> result(input.size());

  cl_platform_id platform;
  clGetPlatformIDs(1, &platform, nullptr);

  cl_device_id device;
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

  cl_context context;
  context = clCreateContextFromType(nullptr, CL_DEVICE_TYPE_GPU, nullptr, nullptr, nullptr);

  cl_command_queue queue;
  cl_queue_properties properties[] = {CL_QUEUE_PROPERTIES, 0, 0};
  queue = clCreateCommandQueueWithProperties(context, device, properties, nullptr);

  cl_mem in, out;
  in = clCreateBuffer(context, CL_MEM_READ_ONLY, input.size() * sizeof(float), nullptr, nullptr);
  out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, input.size() * sizeof(float), nullptr, nullptr);

  clEnqueueWriteBuffer(queue, in, CL_TRUE, 0, input.size() * sizeof(float), input.data(), 0, nullptr, nullptr);

  size_t srclen[] = { strlen(source) };
  cl_program program = clCreateProgramWithSource(context, 1, &source, srclen, nullptr);
  clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
  cl_kernel kernel = clCreateKernel(program, "kernel", nullptr);

  const size_t size = input.size();

  clSetKernelArg(kernel, 0, sizeof(cl_mem), &in);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &out);
  clSetKernelArg(kernel, 2, sizeof(int), &size);

  size_t group;
  clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &group, nullptr);
  clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &size, &group, 0, nullptr, nullptr);
  clFinish(queue);

  clEnqueueReadBuffer(queue, out, CL_TRUE, 0, size * sizeof(float), result.data(), 0, nullptr, nullptr);

  clReleaseMemObject(in);
  clReleaseMemObject(out);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  return result;
}