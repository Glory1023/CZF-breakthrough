#include <catch2/catch.hpp>

#include "czf/env/czf_env/gomoku/gomoku.h"
using czf::env::czf_env::gomoku::GomokuGame;

TEST_CASE("gomoku: name", "[GomokuGame]") {
  REQUIRE(GomokuGame{}.name() == "gomoku");
}

TEST_CASE("gomoku: is_terminal", "[GomokuState]") {
  auto game = std::make_shared<GomokuGame>();
  auto state = game->new_initial_state();
  REQUIRE(!state->is_terminal());
}

TEST_CASE("gomoku: serialize & deserialize_state", "[GomokuGame]") {
  auto game = std::make_shared<GomokuGame>();
  auto state = game->new_initial_state();

  while (!state->is_terminal()) {
    auto action = state->legal_actions()[0];
    state->apply_action(action);
    auto state2 = game->deserialize_state(state->serialize());
    REQUIRE(state->serialize() == state2->serialize());
  }
  REQUIRE(state->is_terminal());
}

TEST_CASE("gomoku: observation_tensor", "[GomokuGame]") {
  auto game = std::make_shared<GomokuGame>();
  auto state = game->new_initial_state();

  auto shape = game->observation_tensor_shape();
  auto size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
  auto tensor = state->observation_tensor();
  REQUIRE(size == tensor.size());
}
