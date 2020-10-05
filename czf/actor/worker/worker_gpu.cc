#include <chrono>
#include <iostream>
#include <thread>

#include "worker.h"

namespace czf::actor::worker {
using namespace std::chrono_literals;

void WorkerGPU::run() {
  while (Worker::running_) {
    // std::cout << "[GPU] " << std::this_thread::get_id() << std::endl;
    std::this_thread::sleep_for(7s);
  }
}

}  // namespace czf::actor::worker
