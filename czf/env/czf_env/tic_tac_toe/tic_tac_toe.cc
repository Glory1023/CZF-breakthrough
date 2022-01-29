#include "czf/env/czf_env/tic_tac_toe/tic_tac_toe.h"

#include <numeric>
#include <sstream>

namespace czf::env::czf_env::tic_tac_toe {

TicTacToeState::TicTacToeState(GamePtr game_ptr)
    : State(std::move(game_ptr)), turn_(0), winner_(-1) {
  board_.fill(2);
}

StatePtr TicTacToeState::clone() const { return std::make_unique<TicTacToeState>(*this); }

void TicTacToeState::apply_action(const Action& action) {
  Player player = current_player();
  board_[action] = player;
  ++turn_;
  if (has_line(player)) {
    winner_ = player;
  }
  history_.push_back(action);
}

std::vector<Action> TicTacToeState::legal_actions() const {
  std::vector<Action> actions;
  actions.reserve(9 - turn_);
  for (int i = 0; i < 9; ++i) {
    if (board_[i] == 2) {
      actions.push_back(i);
    }
  }
  return actions;
}

std::string TicTacToeState::to_string() const {
  std::string c;
  for (const auto& b : board_) c += "OX "[b];
  std::stringstream ss;
  ss << "┌───┬───┬───┐" << std::endl;
  ss << "│ " << c[0] << " │ " << c[1] << " │ " << c[2] << " │" << std::endl;
  ss << "├───┼───┼───┤" << std::endl;
  ss << "│ " << c[3] << " │ " << c[4] << " │ " << c[5] << " │" << std::endl;
  ss << "├───┼───┼───┤" << std::endl;
  ss << "│ " << c[6] << " │ " << c[7] << " │ " << c[8] << " │" << std::endl;
  ss << "└───┴───┴───┘";
  return ss.str();
}

bool TicTacToeState::is_terminal() const { return turn_ == 9 || winner_ != -1; }

Player TicTacToeState::current_player() const { return turn_ % 2; }

std::vector<float> TicTacToeState::observation_tensor() const {
  std::vector<float> tensor;
  auto shape = game()->observation_tensor_shape();
  auto size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
  tensor.reserve(size);

  for (int i = 0; i < 9; ++i) {
    tensor.push_back(static_cast<float>(board_[i] == 0));
  }
  for (int i = 0; i < 9; ++i) {
    tensor.push_back(static_cast<float>(board_[i] == 1));
  }
  for (int i = 0; i < 9; ++i) {
    tensor.push_back(static_cast<float>(board_[i] == 2));
  }
  Player player = current_player();
  for (int i = 0; i < 9; ++i) {
    tensor.push_back(static_cast<float>(player));
  }

  return tensor;
}

bool TicTacToeState::has_line(const Player& player) const {
  int b = player;
  return (board_[0] == b && board_[1] == b && board_[2] == b) ||
         (board_[3] == b && board_[4] == b && board_[5] == b) ||
         (board_[6] == b && board_[7] == b && board_[8] == b) ||
         (board_[0] == b && board_[3] == b && board_[6] == b) ||
         (board_[1] == b && board_[4] == b && board_[7] == b) ||
         (board_[2] == b && board_[5] == b && board_[8] == b) ||
         (board_[0] == b && board_[4] == b && board_[8] == b) ||
         (board_[2] == b && board_[4] == b && board_[6] == b);
}

std::string TicTacToeState::serialize() const {
  std::stringstream ss;
  for (const Action action : history_) {
    ss << action << " ";
  }
  return ss.str();
}

std::vector<float> TicTacToeState::rewards() const {
  if (winner_ == -1) {
    return {0.0F, 0.0F};
  }
  if (winner_ == 0) {
    return {1.0F, -1.0F};
  }
  // if (winner_ == 1)
  return {-1.0F, 1.0F};
}

std::string TicTacToeGame::name() const { return "tic_tac_toe"; }
int TicTacToeGame::num_players() const { return 2; }
int TicTacToeGame::num_distinct_actions() const { return 9; }
std::vector<int> TicTacToeGame::observation_tensor_shape() const { return {4, 3, 3}; }

StatePtr TicTacToeGame::new_initial_state() const {
  return std::make_unique<TicTacToeState>(shared_from_this());
}

int TicTacToeGame::num_transformations() const { return 8; }

std::vector<float> TicTacToeGame::transform_observation(const std::vector<float>& observation,
                                                        int type) const {
  std::vector<float> transformed_observation(observation);
  transform(&transformed_observation[0], type);
  transform(&transformed_observation[9], type);
  transform(&transformed_observation[18], type);
  return transformed_observation;
}

std::vector<float> TicTacToeGame::transform_policy(const std::vector<float>& policy,
                                                   int type) const {
  std::vector<float> transformed_policy(policy);
  transform(&transformed_policy[0], type);
  return transformed_policy;
}

std::vector<float> TicTacToeGame::restore_policy(const std::vector<float>& policy,
                                                 int type) const {
  std::vector<float> restored_policy(policy);
  int restored_type;
  switch (type) {
    default:
    case 0:
      restored_type = 0;
      break;
    case 1:
      restored_type = 3;
      break;
    case 2:
      restored_type = 2;
      break;
    case 3:
      restored_type = 1;
      break;
    case 4:
      restored_type = 4;
      break;
    case 5:
      restored_type = 5;
      break;
    case 6:
      restored_type = 6;
      break;
    case 7:
      restored_type = 7;
      break;
  }
  transform(&restored_policy[0], restored_type);
  return restored_policy;
}

void TicTacToeGame::transform(float* tensor, int type) const {
  switch (type) {
    default:
    case 0:
      break;
    case 1:
      rotate_right(tensor);
      break;
    case 2:
      reverse(tensor);
      break;
    case 3:
      rotate_left(tensor);
      break;
    case 4:
      transpose(tensor);
      break;
    case 5:
      rotate_right(tensor);
      transpose(tensor);
      break;
    case 6:
      reverse(tensor);
      transpose(tensor);
      break;
    case 7:
      rotate_left(tensor);
      transpose(tensor);
      break;
  }
}

void TicTacToeGame::rotate_right(float* tensor) const {
  transpose(tensor);
  reflect_horizontal(tensor);
}

void TicTacToeGame::rotate_left(float* tensor) const {
  transpose(tensor);
  reflect_vertical(tensor);
}

void TicTacToeGame::reverse(float* tensor) const {
  reflect_horizontal(tensor);
  reflect_vertical(tensor);
}

void TicTacToeGame::transpose(float* tensor) const {
  std::swap(tensor[1], tensor[3]);
  std::swap(tensor[2], tensor[6]);
  std::swap(tensor[5], tensor[7]);
}

void TicTacToeGame::reflect_horizontal(float* tensor) const {
  std::swap(tensor[0], tensor[2]);
  std::swap(tensor[3], tensor[5]);
  std::swap(tensor[6], tensor[8]);
}

void TicTacToeGame::reflect_vertical(float* tensor) const {
  std::swap(tensor[0], tensor[6]);
  std::swap(tensor[1], tensor[7]);
  std::swap(tensor[2], tensor[8]);
}

StatePtr TicTacToeGame::deserialize_state(const std::string& str) const {
  std::stringstream ss(str);
  int action;
  StatePtr state = new_initial_state();
  while (ss >> action) {
    state->apply_action(action);
  }
  return state->clone();
}

}  // namespace czf::env::czf_env::tic_tac_toe
