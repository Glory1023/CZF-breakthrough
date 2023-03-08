#include <catch2/catch.hpp>
#include <iostream>

#include "czf/env/czf_env/breakthrough/breakthrough.h"
using czf::env::czf_env::breakthrough::BreakThroughGame;

TEST_CASE("breakthrough: name", "[BreakThroughGame]") {
  REQUIRE(BreakThroughGame{}.name() == "breakthrough");
}

TEST_CASE("breakthrough: is_terminal", "[BreakThroughState]") {
  auto game = std::make_shared<BreakThroughGame>();
  auto state = game->new_initial_state();
  REQUIRE(!state->is_terminal());
}

TEST_CASE("breakthrough: serialize & deserialize_state", "[BreakThroughGame]") {
  auto game = std::make_shared<BreakThroughGame>();
  auto state = game->new_initial_state();

  while (!state->is_terminal()) {
    auto action = state->legal_actions()[0];
    state->apply_action(action);
    auto state2 = game->deserialize_state(state->serialize());
    REQUIRE(state->serialize() == state2->serialize());
  }
  REQUIRE(state->is_terminal());
}

TEST_CASE("breakthrough: observation_tensor", "[BreakThroughGame]") {
  auto game = std::make_shared<BreakThroughGame>();
  auto state = game->new_initial_state();

  auto shape = game->observation_tensor_shape();
  auto size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
  auto tensor = state->observation_tensor();
  REQUIRE(size == tensor.size());
}
