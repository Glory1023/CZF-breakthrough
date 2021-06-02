#include <catch2/catch.hpp>

#include "czf/env/czf_env/surakarta/surakarta.h"
using czf::env::czf_env::surakarta::SurakartaGame;

TEST_CASE("surakarta: name", "[SurakartaGame]") {
  REQUIRE(SurakartaGame{}.name() == "surakarta");
}

TEST_CASE("surakarta: is_terminal", "[SurakartaState]") {
  auto game = std::make_shared<SurakartaGame>();
  auto state = game->new_initial_state();
  REQUIRE(!state->is_terminal());
}

TEST_CASE("surakarta: serialize & deserialize_state", "[SurakartaGame]") {
  auto game = std::make_shared<SurakartaGame>();
  auto state = game->new_initial_state();

  while (!state->is_terminal()) {
    auto action = state->legal_actions()[0];
    state->apply_action(action);
    auto state2 = game->deserialize_state(state->serialize());
    REQUIRE(state->serialize() == state2->serialize());
  }
  REQUIRE(state->is_terminal());
}