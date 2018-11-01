#include <ATen/core/thread_pool.h>
#include <ATen/core/ivalue.h>

namespace c10 {

void ThreadPool::workOnTasksUntilCompleted(
    c10::intrusive_ptr<ivalue::Future> future) {
  while (!future->completed()) {
    auto task = tasks.front();
    tasks.pop();
    task();
  }
}

ThreadPool global_work_queue = {};

} // namespace c10
