#pragma once
#include <memory>
#include <vector>
#include "c10/util/Optional.h"

#include "torch/csrc/jit/ivalue.h"
#include "torch/csrc/WindowsTorchApiMacro.h"

namespace at {
  class Tensor;
}
namespace c10 {
struct IValue;
}
namespace torch { namespace jit {

// The interpreter run Graphs with Tensor inputs and Tensor outputs
// a separate component in the autograd handles unwrapping and wrapping
// variable objects for use in the interpreter.

struct Node;
struct GraphExecutor;
struct CodeImpl;
struct Graph;
struct Node;
using Stack = std::vector<c10::IValue>;

struct TORCH_API Code {
  Code()
    : pImpl(nullptr) {}
  explicit Code(const std::shared_ptr<Graph>& graph);
  ~Code();

  const std::vector<GraphExecutor*>& grad_executors();

  explicit operator bool() const {
    return pImpl != nullptr;
  }

private:
  std::shared_ptr<CodeImpl> pImpl;
  friend struct InterpreterStateImpl;
  friend std::ostream & operator<<(std::ostream & out, const Code & code);
};

// InterpreterState state that and used to compute a Code
// InterpreterStateImpl should not be used as an interface outside InterpreterState
struct InterpreterStateImpl : c10::intrusive_ptr_target {
  // intrusive_ptr needs the constructor visible
  InterpreterStateImpl(const Code & code);
 private:
  c10::intrusive_ptr<Future> getOrCreateFuture();
  c10::intrusive_ptr<Future> runAsync(Stack& stack);
  void run(Stack& stack);
  c10::intrusive_ptr<InterpreterStateImpl> intrusive_from_this();
  bool runImpl(Stack& stack);

  // pc is critical for the interperter to pick up the progress from suspend
  size_t pc = 0;
  c10::intrusive_ptr<Future> future;
  std::shared_ptr<CodeImpl> function; // keep function alive
  // these are just copies of function to prevent indirections in interpreter
  int * int_data;
  const std::vector<bool> & bool_data;


  // this holds all the tensors for this interpreter run
  // we don't bother minimizing the size of this vector, since the extra
  // memory used by the pointers in this will be small
  // instead we are very aggresive about releasing tensors when they become dead
  // to make sure memory management happens efficiently.

  // We optimize for the case where derivatives are run with retain_graph=False
  // in the case where it is true, then the interpreter and this array get copied
  // if this every becomes a bottleneck then we _should_ consider minimizing the
  // total number or register
  std::vector<IValue> registers;

  // single buffer for input/output calls to ATen functions, so that we do not reallocate
  Stack stack;
  friend struct InterpreterState;
};

struct InterpreterState {
  InterpreterState(const Code & code);
  void run(Stack& stack);
  c10::intrusive_ptr<Future> runAsync(Stack& stack);
  c10::intrusive_ptr<Future> getFuture();
  ~InterpreterState();
private:
  InterpreterState(c10::intrusive_ptr<InterpreterStateImpl> pImpl);
  c10::intrusive_ptr<InterpreterStateImpl> pImpl;
  friend struct InterpreterStateImpl;
};

// Created by wait()
struct Suspend : public std::exception {
  virtual const char* what() const noexcept override {
    return "Suspend";
  }

  explicit Suspend(c10::intrusive_ptr<Future> future_) : future(future_) {}

  c10::intrusive_ptr<Future> future;
};

struct InterpreterContinuation {
  InterpreterContinuation(InterpreterState state_, Stack stack_)
      : state(std::move(state_)), stack(std::move(stack_)) {}

  void operator()(void) {
    state.runAsync(stack);
  }

 private:
  InterpreterState state;
  Stack stack;
};
}}
