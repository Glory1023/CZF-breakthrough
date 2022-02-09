#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "worker_manager.h"

namespace czf::actor::alphazero_worker {
namespace {

namespace py = ::pybind11;
using py::literals::operator""_a;

PYBIND11_MODULE(alphazero_worker, m) {  // NOLINT
  py::class_<WorkerOption>(m, "WorkerOption")
      .def(py::init<>())
      .def_readwrite("seed", &WorkerOption::seed)
      .def_readwrite("timeout_us", &WorkerOption::timeout_us)
      .def_readwrite("batch_size", &WorkerOption::batch_size)
      .def_readwrite("num_sampled_transformations", &WorkerOption::num_sampled_transformations);
  py::class_<TreeOption>(m, "TreeOption")
      .def(py::init<>())
      .def_readwrite("simulation_count", &TreeOption::simulation_count)
      .def_readwrite("c_puct", &TreeOption::c_puct)
      .def_readwrite("dirichlet_alpha", &TreeOption::dirichlet_alpha)
      .def_readwrite("dirichlet_epsilon", &TreeOption::dirichlet_epsilon);
  py::class_<WorkerManager>(m, "WorkerManager")
      .def(py::init<>())
      .def("run", &WorkerManager::run, "num_cpu_worker"_a = 1, "num_gpu_worker"_a = 1,
           "num_gpu"_a = 1)
      .def("terminate", &WorkerManager::terminate)
      .def("enqueue_job_batch", &WorkerManager::enqueue_job_batch, "jobs"_a,
           py::call_guard<py::gil_scoped_release>())
      .def("dequeue_job_batch", &WorkerManager::dequeue_job_batch,
           py::call_guard<py::gil_scoped_release>(), py::return_value_policy::move)
      .def("load_from_bytes", &WorkerManager::load_from_bytes, "bytes"_a)
      .def("load_from_file", &WorkerManager::load_from_file, "path"_a)
      .def("load_game", &WorkerManager::load_game, "name"_a)
      .def_readwrite_static("worker_option", &WorkerManager::worker_option);
}

}  // namespace
}  // namespace czf::actor::alphazero_worker
