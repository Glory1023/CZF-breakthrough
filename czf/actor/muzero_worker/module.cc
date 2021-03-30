#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

#include "worker/worker_manager.h"

namespace czf::actor::muzero_worker {
namespace {

namespace py = ::pybind11;
using py::literals::operator""_a;

PYBIND11_MODULE(muzero_worker, m) {  // NOLINT
  using czf::actor::muzero_worker::GameInfo;
  py::class_<GameInfo>(m, "GameInfo")
      .def(py::init<>())
      .def_readwrite("observation_shape", &GameInfo::observation_shape)
      .def_readwrite("state_shape", &GameInfo::state_shape)
      .def_readwrite("all_actions", &GameInfo::all_actions)
      .def_readwrite("num_actions", &GameInfo::num_actions)
      .def_readwrite("two_player", &GameInfo::is_two_player);
  using czf::actor::muzero_worker::WorkerOption;
  py::class_<WorkerOption>(m, "WorkerOption")
      .def(py::init<>())
      .def_readwrite("seed", &WorkerOption::seed)
      .def_readwrite("timeout_us", &WorkerOption::timeout_us)
      .def_readwrite("batch_size", &WorkerOption::batch_size);
  using czf::actor::muzero_worker::TreeOption;
  py::class_<TreeOption>(m, "TreeOption")
      .def(py::init<>())
      .def_readwrite("simulation_count", &TreeOption::simulation_count)
      .def_readwrite("tree_min_value", &TreeOption::tree_min_value)
      .def_readwrite("tree_max_value", &TreeOption::tree_max_value)
      .def_readwrite("c_puct", &TreeOption::c_puct)
      .def_readwrite("dirichlet_alpha", &TreeOption::dirichlet_alpha)
      .def_readwrite("dirichlet_epsilon", &TreeOption::dirichlet_epsilon)
      .def_readwrite("discount", &TreeOption::discount);
  using czf::actor::muzero_worker::worker::WorkerManager;
  py::class_<WorkerManager>(m, "WorkerManager")
      .def(py::init<>())
      .def("run", &WorkerManager::run, "num_cpu_worker"_a = 1,
           "num_gpu_worker"_a = 1, "num_gpu_root_worker"_a = 1, "num_gpu"_a = 1)
      .def("terminate", &WorkerManager::terminate)
      .def("enqueue_job_batch", &WorkerManager::enqueue_job_batch, "jobs"_a,
           py::call_guard<py::gil_scoped_release>())
      .def("wait_dequeue_result", &WorkerManager::wait_dequeue_result,
           py::call_guard<py::gil_scoped_release>(),
           py::return_value_policy::move)
      .def("load_from_bytes", &WorkerManager::load_from_bytes, "bytes"_a)
      .def("load_from_file", &WorkerManager::load_from_file, "path"_a)
      .def_readwrite_static("worker_option", &WorkerManager::worker_option)
      .def_readwrite_static("game_info", &WorkerManager::game_info);
}

}  // namespace
}  // namespace czf::actor::muzero_worker
