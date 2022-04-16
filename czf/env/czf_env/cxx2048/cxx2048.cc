#include "czf/env/czf_env/cxx2048/cxx2048.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

namespace czf::env::czf_env::cxx2048 {

Cxx2048State::Cxx2048State(GamePtr game_ptr) : State(std::move(game_ptr)), turn_(0) {
  board_ = board();
  history_.clear();
}

Player Cxx2048State::current_player() const { return 0; }

std::vector<Action> Cxx2048State::legal_actions() const {
  std::vector<Action> ret;
  for (int op : {0, 1, 2, 3}) {
    int reward = board(board_).slide(op);
    if (reward != -1) {
      ret.push_back(op);
    }
  }
  return ret;
}

std::vector<float> Cxx2048State::observation_tensor() const {
  auto shape = game()->observation_tensor_shape();
  auto size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
  std::vector<float> tensor(size, 0);
  for (int pos = 0; pos < kNumOfGrids; pos++) {
    tensor[board_(pos) * kNumOfGrids + pos] = 1;
  }
  return tensor;
}

std::vector<float> Cxx2048State::feature_tensor() const { return observation_tensor(); }

void Cxx2048State::apply_action(const Action& action) {
  if (is_chance_node()) {
    unsigned pos = action / 2;
    uint32_t cell = 1 + (action % 2);
    int reward = board_.place(pos, cell);
    if (reward == -1) {
      std::cerr << "Error [apply chance]" << std::endl;
    }
    turn_++;
    history_.push_back(action);
  } else {
    reward_ = board_.slide(action);
    if (reward_ == -1) {
      std::cerr << "Error [apply action]" << std::endl;
    }
    turn_++;
    history_.push_back(action);
  }
}

bool Cxx2048State::is_terminal() const { return legal_actions().empty(); }

std::vector<float> Cxx2048State::rewards() const { return {(float)reward_}; }

StatePtr Cxx2048State::clone() const { return std::make_unique<Cxx2048State>(*this); }

std::string Cxx2048State::to_string() const {
  std::stringstream ss;
  ss << board_;
  return ss.str();
}

std::string Cxx2048State::serialize() const {
  std::stringstream ss;
  for (const Action action : history_) {
    ss << action << " ";
  }
  return ss.str();
}

bool Cxx2048State::is_chance_node() const {
  if (turn_ < 2) {
    return true;
  } else if (turn_ % 2 == 1 && !is_terminal()) {
    return true;
  }
  return false;
}

std::vector<std::pair<Action, float>> Cxx2048State::legal_chance_outcome_probs() const {
  std::vector<std::pair<Action, float>> ret;
  if (is_chance_node()) {
    std::vector<int> space;
    for (int pos = 0; pos < kNumOfGrids; pos++) {
      if (board_(pos) == 0) {
        space.push_back(pos);
      }
    }
    for (int pos : space) {
      ret.emplace_back(pos * 2, 0.9 / space.size());
      ret.emplace_back(pos * 2 + 1, 0.1 / space.size());
    }
  }
  return ret;
}

std::string Cxx2048Game::name() const { return "2048"; }
int Cxx2048Game::num_players() const { return 1; }
int Cxx2048Game::num_distinct_actions() const { return 4; }
std::vector<int> Cxx2048Game::observation_tensor_shape() const {
  return {17, kBoardSize, kBoardSize};
}

StatePtr Cxx2048Game::new_initial_state() const {
  return std::make_unique<Cxx2048State>(shared_from_this());
}

StatePtr Cxx2048Game::deserialize_state(const std::string& str) const {
  std::stringstream ss(str);
  int action;
  StatePtr state = new_initial_state();
  while (ss >> action) {
    state->apply_action(action);
  }
  return state->clone();
}

int Cxx2048Game::num_chance_outcomes() const { return 32; };

}  // namespace czf::env::czf_env::cxx2048