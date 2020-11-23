#include <iostream>
#include <fstream>
#include <chrono>
#include "CL/cl.h"

#define MAX_SOURCE_SIZE (0x100000)

using namespace std;

int main() {
    const int PRIME_LIST_SIZE = 1024 * 1024 * 64;
    const int WORK_GROUP_SIZE = 128;
    const auto LIST_MEM_SIZE = sizeof(int) * PRIME_LIST_SIZE;
    auto *IN = (int *) malloc(LIST_MEM_SIZE);

    for (int i = 0; i < PRIME_LIST_SIZE; i++) {
        IN[i] = (i*2) + 1;
    }

    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen("./prime_check_kernel.cl", "r");

    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }

    source_str = (char *) malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    // Get platform and device information
    cl_platform_id platform_id = nullptr;
    cl_device_id device_id = nullptr;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1,
                         &device_id, &ret_num_devices);

    cl_context context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &ret);

    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, nullptr, &ret);

    cl_mem in_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                      LIST_MEM_SIZE, nullptr, &ret);
    cl_mem out_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(bool) * PRIME_LIST_SIZE, nullptr, &ret);

    ret = clEnqueueWriteBuffer(command_queue, in_mem_obj, CL_TRUE, 0,
                               LIST_MEM_SIZE, IN, 0, nullptr, nullptr);

    cl_program program = clCreateProgramWithSource(context, 1,
                                                   (const char **) &source_str, (const size_t *) &source_size, &ret);

    ret = clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);

    if (ret != CL_SUCCESS) {
        size_t log_size;
        char *program_log;

        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        program_log = (char*) malloc(log_size+1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
                              log_size+1, program_log, nullptr);
        printf("\n=== ERROR ===\n\n%s\n=============\n", program_log);
        free(program_log);
        exit(1);
    }

    cl_kernel kernel = clCreateKernel(program, "check_prime", &ret);

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &in_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) & out_mem_obj);


    auto start = chrono::high_resolution_clock::now();
    size_t global_item_size = PRIME_LIST_SIZE;
    size_t local_item_size = WORK_GROUP_SIZE;
    printf("Enqueueing %d prime checks with a work group size of %d\n", PRIME_LIST_SIZE, WORK_GROUP_SIZE);
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, nullptr,
                                 &global_item_size, &local_item_size, 0, nullptr, nullptr);

    auto *OUT = (bool *) malloc(sizeof(bool) * PRIME_LIST_SIZE);
    ret = clEnqueueReadBuffer(command_queue, out_mem_obj, CL_TRUE, 0,
                              sizeof(bool) * PRIME_LIST_SIZE, OUT, 0, nullptr, nullptr);
    auto finish = chrono::high_resolution_clock::now();

    printf("Writing primes to file...\n");
    ofstream prime_file;
    prime_file.open("primes.txt");

    int count = 0;
    for (int i = 0; i < PRIME_LIST_SIZE; i++) {
        if (OUT[i]) {
            prime_file << IN[i] << "\n";
            count++;
        }
    }
    prime_file.close();
    chrono::duration<double> elapsed = finish - start;

    printf("Calculated %d primes in %f ms\n", count, elapsed.count() * 1000);

    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(in_mem_obj);
    ret = clReleaseMemObject(out_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(IN);
    free(OUT);
    return 0;
}