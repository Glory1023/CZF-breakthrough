#include <catch2/catch.hpp>

#include "czf/env/czf_env/tic_tac_toe/tic_tac_toe.h"
using czf::env::czf_env::tic_tac_toe::TicTacToeGame;

TEST_CASE("tic_tac_toe: name", "[TicTacToeGame]") {
  REQUIRE(TicTacToeGame{}.name() == "tic_tac_toe");
}

TEST_CASE("tic_tac_toe: is_terminal", "[TicTacToeState]") {
  auto game = std::make_shared<TicTacToeGame>();
  auto state = game->new_initial_state();
  REQUIRE(!state->is_terminal());
}

TEST_CASE("tic_tac_toe: serialize & deserialize_state", "[TicTacToeGame]") {
  auto game = std::make_shared<TicTacToeGame>();
  auto state = game->new_initial_state();

  while (!state->is_terminal()) {
    auto action = state->legal_actions()[0];
    state->apply_action(action);
    auto state2 = game->deserialize_state(state->serialize());
    REQUIRE(state->serialize() == state2->serialize());
  }
  REQUIRE(state->is_terminal());
}