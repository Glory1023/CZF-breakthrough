#include <pybind11/pybind11.h>

#include <memory>

#include "worker/worker.h"

namespace czf {
namespace {

namespace py = ::pybind11;
PYBIND11_MODULE(worker, m) {           // NOLINT
  using namespace czf::actor::worker;  // NOLINT
  py::class_<WorkerManager>(m, "WorkerManager")
      .def(py::init<>())
      .def("register_worker", &WorkerManager::register_worker,
           py::keep_alive<1, 2>())
      .def("enqueue_job", &WorkerManager::enqueue_job)
      .def("wait_dequeue_result", &WorkerManager::wait_dequeue_result,
           py::call_guard<py::gil_scoped_release>())
      .def("run", &WorkerManager::run)
      .def("terminate", &WorkerManager::terminate);
  py::class_<Worker, std::shared_ptr<Worker>>(m, "Worker");

  py::class_<WorkerCPU, Worker, std::shared_ptr<WorkerCPU>>(m, "WorkerCPU")
      .def(py::init<>())
      .def("run", &WorkerCPU::run)
      .def("terminate", &WorkerCPU::terminate);
  py::class_<WorkerGPU, Worker, std::shared_ptr<WorkerGPU>>(m, "WorkerGPU")
      .def(py::init<>())
      .def("run", &WorkerGPU::run)
      .def("terminate", &WorkerGPU::terminate);
}

}  // namespace
}  // namespace czf
