#include <catch2/catch.hpp>

#include "czf/env/czf_env/surakarta_split_action/surakarta_split_action.h"
using czf::env::czf_env::surakarta::SurakartaSplitActionGame;

TEST_CASE("surakarta_split_action: name", "[SurakartaSplitActionGame]") {
  REQUIRE(SurakartaSplitActionGame{}.name() == "surakarta_split_action");
}

TEST_CASE("surakarta_split_action: is_terminal",
          "[SurakartaSplitActionState]") {
  auto game = std::make_shared<SurakartaSplitActionGame>();
  auto state = game->new_initial_state();
  REQUIRE(!state->is_terminal());
}

TEST_CASE("surakarta_split_action: serialize & deserialize_state",
          "[SurakartaSplitActionGame]") {
  auto game = std::make_shared<SurakartaSplitActionGame>();
  auto state = game->new_initial_state();

  while (!state->is_terminal()) {
    auto action = state->legal_actions()[0];
    state->apply_action(action);
    auto state2 = game->deserialize_state(state->serialize());
    REQUIRE(state->serialize() == state2->serialize());
  }
  REQUIRE(state->is_terminal());
}

TEST_CASE("surakarta_split_action: observation_tensor",
          "[SurakartaSplitActionGame]") {
  auto game = std::make_shared<SurakartaSplitActionGame>();
  auto state = game->new_initial_state();

  auto shape = game->observation_tensor_shape();
  auto size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
  auto tensor = state->observation_tensor();
  REQUIRE(size == tensor.size());
}
