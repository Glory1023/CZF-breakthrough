#include <pybind11/pybind11.h>

#include <memory>

#include "worker/worker.h"

namespace czf {
namespace {

namespace py = ::pybind11;
using py::literals::operator""_a;

PYBIND11_MODULE(worker, m) {  // NOLINT
  using czf::actor::worker::WorkerManager;
  py::class_<WorkerManager>(m, "WorkerManager")
      .def(py::init<>())
      .def("run", &WorkerManager::run, "num_cpu_worker"_a, "num_gpu_worker"_a)
      .def("terminate", &WorkerManager::terminate)
      .def("enqueue_job", &WorkerManager::enqueue_job, "job"_a, "observation"_a,
           "observation_shape"_a, "legal_actions"_a)
      .def("wait_dequeue_result", &WorkerManager::wait_dequeue_result,
           py::call_guard<py::gil_scoped_release>())
      .def("load_model", &WorkerManager::load_model, "path"_a);
}

}  // namespace
}  // namespace czf
