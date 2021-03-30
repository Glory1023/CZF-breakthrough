#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "czf/env/czf_env/game.h"

namespace czf::env::czf_env {
namespace {

namespace py = ::pybind11;
using py::literals::operator""_a;

PYBIND11_MODULE(czf_env, m) {  // NOLINT
  m.def("load_game", &load_game);
  m.def("available_games", &available_games);

  py::class_<State>(m, "State")
      .def_property_readonly("game", &State::game)
      .def_property_readonly("current_player", &State::current_player)
      .def_property_readonly("legal_actions", &State::legal_actions)
      .def_property_readonly("observation_tensor", &State::observation_tensor)
      .def_property_readonly("feature_tensor", &State::feature_tensor)
      .def("apply_action", &State::apply_action)
      .def_property_readonly("is_terminal", &State::is_terminal)
      .def_property_readonly("rewards", &State::rewards)
      .def("clone", &State::clone)
      .def("serialize", &State::serialize)
      .def("__repr__", &State::to_string);

  py::class_<Game, std::shared_ptr<Game>>(m, "Game")
      .def_property_readonly("name", &Game::name)
      .def_property_readonly("num_players", &Game::num_players)
      .def_property_readonly("num_distinct_actions",
                             &Game::num_distinct_actions)
      .def_property_readonly("observation_tensor_shape",
                             &Game::observation_tensor_shape)
      .def("new_initial_state", &Game::new_initial_state)
      .def_property_readonly("num_transformations", &Game::num_transformations)
      .def("transform_observation", &Game::transform_observation)
      .def("transform_policy", &Game::transform_policy)
      .def("restore_policy", &Game::restore_policy)
      .def("action_to_string", &Game::action_to_string)
      .def("string_to_action", &Game::string_to_action)
      .def("deserialize_state", &Game::deserialize_state);
}

}  // namespace
}  // namespace czf::env::czf_env
