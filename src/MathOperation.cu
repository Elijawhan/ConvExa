#include <MathOperation.h>

#include <helper_cuda.h>

///////// this is where a cuda or host specific version would go... //////////
///////// cuda included here for reference...                       //////////

namespace kernels {

__global__ void do_a_thing() {
  printf("Hello, world, from block %d, thread %d!\n", blockIdx.x, threadIdx.x);
}

} // end namespace kernels 

// struct or class is fine... this isn't exposed so opted for struct
// for simplicity
struct MathOperation::Implementation {
  Implementation(int size_of_operation) : 
    size_of_operation(size_of_operation),
    byte_size(size_of_operation * sizeof(int)) {
    // put any cudaMallocs in here
    checkCudaErrors(cudaMalloc(&temporary_buffer, byte_size));
    // also plan the kernel execution sizes, etc.
  }
  ~Implementation() {
    // put any cudaFrees you need in here
    checkCudaErrors(cudaFree(temporary_buffer));
  }
  // obey rule of five explicitly
  Implementation & operator=(Implementation const &) = delete;
  Implementation & operator=(Implementation &&) = delete;
  Implementation(Implementation const &) = delete;
  Implementation(Implementation &&) = delete;

  // this is where we put the execution stuff
  void operator() (
    float * output_buffer,
    float const * const input_buffer    
  ) {
    checkCudaErrors(cudaMemcpy(temporary_buffer, input_buffer, byte_size,
          cudaMemcpyDeviceToDevice));

    kernels::do_a_thing<<<1, 1>>>();
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMemcpy(output_buffer, temporary_buffer, byte_size,
          cudaMemcpyDeviceToDevice));

  }

  // nice facts to know
  int const size_of_operation;
  size_t const byte_size;
  
  // workspace buffers
  float * temporary_buffer;

};

//////////// should look pretty much the same between host and device ////////

// constructor implementation
MathOperation::MathOperation(int size_of_operation) :
  implementation_(std::make_shared<Implementation>(size_of_operation)) {}

// operator implementation
void MathOperation::operator() (
    float * output_buffer,
    float const * const input_buffer
) {
  // dereference and call
  (*implementation_)(
    output_buffer,
    input_buffer
  );
}
