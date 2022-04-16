#pragma once

#include <vector>

#include "czf/env/czf_env/cxx2048/board.h"
#include "czf/env/czf_env/game.h"

namespace czf::env::czf_env::cxx2048 {

constexpr int kBoardSize = 4;
constexpr int kNumOfGrids = kBoardSize * kBoardSize;

class Cxx2048State final : public State {
 public:
  Cxx2048State(GamePtr);
  Cxx2048State(const Cxx2048State &) = default;

  Player current_player() const override;
  std::vector<Action> legal_actions() const override;
  std::vector<float> observation_tensor() const override;
  std::vector<float> feature_tensor() const override;
  void apply_action(const Action &) override;
  bool is_terminal() const override;
  std::vector<float> rewards() const override;
  StatePtr clone() const override;
  std::string to_string() const override;

  std::string serialize() const override;

  bool is_chance_node() const override;
  std::vector<std::pair<Action, float>> legal_chance_outcome_probs() const override;

 private:
  board board_;
  int turn_;
  int reward_;
  std::vector<Action> history_;
};

class Cxx2048Game final : public Game {
 public:
  std::string name() const override;
  int num_players() const override;
  int num_distinct_actions() const override;
  StatePtr new_initial_state() const override;
  std::vector<int> observation_tensor_shape() const override;

  StatePtr deserialize_state(const std::string &) const override;

  int num_chance_outcomes() const override;
};

// *IMPORTANT* Register this game to the factory
namespace {
Registration<Cxx2048Game> registration;
}
}  // namespace czf::env::czf_env::cxx2048