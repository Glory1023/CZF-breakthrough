#include <pybind11/pybind11.h>

#include <memory>

#include "worker/worker.hpp"

namespace czf {
namespace {

namespace py = ::pybind11;
PYBIND11_MODULE(worker, m) {  // NOLINT
  using namespace czf::workers;
  py::class_<WorkerManager>(m, "WorkerManager")
      .def(py::init<>())
      .def("register_worker", &WorkerManager::register_worker,
           py::keep_alive<1, 2>())
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
