#pragma once

#include <array>

#include "czf/env/czf_env/game.h"

namespace czf::env::czf_env::tic_tac_toe {

class TicTacToeState final : public State {
 public:
  TicTacToeState(GamePtr);
  TicTacToeState(const TicTacToeState&) = default;
  StatePtr clone() const override;
  void apply_action(const Action&) override;
  std::vector<Action> legal_actions() const override;
  bool is_terminal() const override;

  Player current_player() const override;
  std::vector<float> rewards() const override;
  std::vector<float> observation_tensor() const override;
  std::string to_string() const override;

  bool has_line(const Player&) const;

  std::string serialize() const override;

 private:
  // 0 -> O
  // 1 -> X
  // 2 -> empty
  std::array<short, 9> board_;
  int turn_;
  Player winner_;
  std::vector<Action> history_;
};

class TicTacToeGame final : public Game {
 public:
  std::string name() const override;
  int num_players() const override;
  int num_distinct_actions() const override;
  StatePtr new_initial_state() const override;
  std::vector<int> observation_tensor_shape() const override;

  int num_transformations() const override;
  std::vector<float> transform_observation(const std::vector<float>&,
                                           int) const override;
  std::vector<float> transform_policy(const std::vector<float>&,
                                      int) const override;
  std::vector<float> restore_policy(const std::vector<float>&,
                                    int) const override;

  StatePtr deserialize_state(const std::string& str = "") const override;

 private:
  void transform(float*, int) const;
  void rotate_right(float*) const;
  void rotate_left(float*) const;
  void reverse(float*) const;
  void transpose(float*) const;
  void reflect_horizontal(float*) const;
  void reflect_vertical(float*) const;
};

// *IMPORTANT* Register this game to the factory
namespace {
Registration<TicTacToeGame> registration;
}

}  // namespace czf::env::czf_env::tic_tac_toe
