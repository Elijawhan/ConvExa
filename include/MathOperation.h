#ifndef __mathoperation_h
#define __mathoperation_h

#include <memory>

class MathOperation {
public:
  // make a mathematical functor to do an operation of a specific size
  MathOperation(int size_of_operation);

  // do the operation passing an input buffer and an output buffer
  void operator()(
    float * output_buffer,
    float const * const input_buffer    
  ) ;
private:
  // this is called the pointer to implementation pattern (Pimpl pattern)
  class Implementation;
  std::shared_ptr<Implementation> implementation_;
};

#endif
