#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "czf.h"

namespace czf {
namespace {
namespace py = ::pybind11;
PYBIND11_MODULE(pyczf, m) {
  py::class_<State> state(m, "State");
  state.def("apply_action", &State::ApplyAction)
      .def("current_player", &State::CurrentPlayer)
      .def("get_game", &State::GetGame)
      .def("is_terminal", &State::IsTerminal)
      .def("legal_actions", &State::LegalActions)
      .def("observation_tensor", &State::ObservationTensor)
      .def("serialize", &State::Serialize);

  py::class_<Game, std::shared_ptr<Game>> game(m, "Game");
  game.def("short_name", &Game::ShortName)
      .def("name", &Game::Name)
      .def("num_distinct_actions", &Game::NumDistinctActions)
      .def("new_initial_state", &Game::NewInitialState)
      .def("deserialize_state", &Game::DeserializeState)
      .def("observation_tensor_shape", &Game::ObservationTensorShape)
      .def("policy_tensor_shape", &Game::PolicyTensorShape);

  m.def("load_game", &czf::LoadGame);
}
}  // namespace
}  // namespace czf