#include "czf/env/czf_env/surakarta_split_action/surakarta_split_action.h"

#include <algorithm>
#include <iomanip>
#include <numeric>
#include <sstream>

namespace czf::env::czf_env::surakarta {

SurakartaSplitActionState::SurakartaSplitActionState(GamePtr game_ptr)
    : State(std::move(game_ptr)),
      turn_(0),
      winner_(-1),
      black_piece_(12),
      white_piece_(12),
      without_capture_turn_(0) {
  std::fill(std::begin(board_), std::end(board_), EMPTY);
  std::fill(std::begin(board_), std::begin(board_) + 12, WHITE);
  std::fill(std::end(board_) - 12, std::end(board_), BLACK);
  repeat_.clear();
  repeat_[board_]++;
  history_.clear();
}

StatePtr SurakartaSplitActionState::clone() const {
  return std::make_unique<SurakartaSplitActionState>(*this);
}

void SurakartaSplitActionState::apply_action(const Action &action) {
  if (turn_ % 2 == 0) {  // choose
    ++turn_;
    history_.push_back(action);
  } else {  // move and capture
    Player player = current_player();
    const int src = history_[history_.size() - 1];
    if (board_[action] != EMPTY) {
      without_capture_turn_ = 0;
      repeat_.clear();
      if (board_[action] == BLACK) {
        black_piece_--;
      } else if (board_[action] == WHITE) {
        white_piece_--;
      }
    } else {
      without_capture_turn_++;
    }
    std::swap(board_[src], board_[action]);
    board_[src] = EMPTY;
    ++turn_;
    history_.push_back(action);
    repeat_[board_]++;

    if (!has_piece(1 - player)) {
      winner_ = player;
    } else if (repeat_[board_] >= kMaxRepeatBoard || without_capture_turn_ >= kMaxNoCaptureTurn ||
               turn_ >= kMaxTurn) {
      if (black_piece_ > white_piece_) {
        winner_ = BLACK;
      } else if (black_piece_ < white_piece_) {
        winner_ = WHITE;
      } else if (black_piece_ == white_piece_) {
        winner_ = EMPTY;
      }
    }
  }
}

std::vector<Action> SurakartaSplitActionState::legal_actions() const {
  std::vector<Action> actions;
  for (uint32_t i = 0; i < kPolicyDim; i++) {
    if (is_legal_action(i)) {
      actions.push_back(i);
    }
  }
  return actions;
}

bool SurakartaSplitActionState::is_legal_action(const Action &action) const {
  if (action >= kPolicyDim || action < 0) return false;
  const Player player = current_player();
  if (turn_ % 2 == 0) {  // choose
    if (board_[action] == player && (have_move(action) || have_capture(action))) {
      return true;
    }
  } else {  // move and capture
    const Action &src = history_[history_.size() - 1];
    if (board_[action] == EMPTY) {  // move
      if ((std::abs(action / kBoardSize - src / kBoardSize) <= 1) &&
          (std::abs(action % kBoardSize - src % kBoardSize) <= 1))
        return true;
    } else if (board_[action] == 1 - player) {  // capture
      if ((kIsOuter[action] && kIsOuter[src]) &&
          (is_legal_capture_this_circle(action, src, kOuter) ||
           is_legal_capture_this_circle(action, src, kOuterReverse))) {
        return true;
      } else if ((kIsInter[action] && kIsInter[src]) &&
                 (is_legal_capture_this_circle(action, src, kInter) ||
                  is_legal_capture_this_circle(action, src, kInterReverse))) {
        return true;
      }
    }
  }
  return false;
}

bool SurakartaSplitActionState::is_legal_capture_this_circle(
    const Action &action, const Action &src, const std::array<int, 56> &circle) const {
  for (int index = 0; index < 28; index++) {
    if (circle[index] == src) {
      bool cancapture = false;
      for (int step = 0; step <= 28; step++) {
        int dest = circle[index + step];
        if (dest == -1)
          cancapture = true;
        else if (dest == src)
          continue;
        else if (board_[dest] != EMPTY) {
          if (dest == action && cancapture) {
            return true;
          }
          break;
        }
      }
    }
  }
  return false;
  return false;
}

bool SurakartaSplitActionState::have_move(const Action &action) const {
  for (int const &diff : MoveDirection) {
    int dest = action + diff;
    if ((dest >= 0 && dest < 36 && board_[dest] == EMPTY) &&
        (std::abs(dest / kBoardSize - action / kBoardSize) <= 1) &&
        (std::abs(dest % kBoardSize - action % kBoardSize) <= 1)) {
      return true;
    }
  }
  return false;
}

bool SurakartaSplitActionState::have_capture(const Action &action) const {
  if (kIsInter[action] && (have_capture_this_circle(action, kInter) ||
                           have_capture_this_circle(action, kInterReverse))) {
    return true;
  }
  if (kIsOuter[action] && (have_capture_this_circle(action, kOuter) ||
                           have_capture_this_circle(action, kOuterReverse))) {
    return true;
  }
  return false;
}

bool SurakartaSplitActionState::have_capture_this_circle(const Action &action,
                                                         const std::array<int, 56> &circle) const {
  for (int index = 0; index < 28; index++) {
    if (circle[index] == action) {
      bool cancapture = false;
      for (int step = 0; step <= 28; step++) {
        int dest = circle[index + step];
        if (dest == -1)
          cancapture = true;
        else if (dest == action)
          continue;
        else if (board_[dest] != EMPTY) {
          if (board_[dest] == 1 - current_player() && cancapture) {
            return true;
          }
          break;
        }
      }
    }
  }
  return false;
}

std::string SurakartaSplitActionState::to_string() const {
  std::stringstream ss;
  const std::vector<std::string> chess{"B", "W", "·"};
  const std::vector<std::string> choose{"b", "w", "·"};

  for (int i = 0; i < kBoardSize; ++i) {
    ss << std::setw(2) << std::setfill(' ') << kBoardSize - i;
    for (int j = 0; j < kBoardSize; ++j) {
      if (turn_ % 2 && history_[history_.size() - 1] == board_[i * kBoardSize + j]) {
        ss << ' ' << choose[board_[i * kBoardSize + j]];
      } else {
        ss << ' ' << chess[board_[i * kBoardSize + j]];
      }
    }
    ss << std::endl;
  }
  ss << "  ";
  for (int i = 0; i < kBoardSize; ++i) ss << ' ' << static_cast<char>('A' + i);
  return ss.str();
}

bool SurakartaSplitActionState::is_terminal() const {
  return (winner_ != -1) || (legal_actions().empty());
}

Player SurakartaSplitActionState::current_player() const { return (Player)((turn_ % 4) / 2); }

std::vector<float> SurakartaSplitActionState::observation_tensor() const {
  std::vector<float> tensor;
  tensor.reserve(8 * 6 * 6);
  Player player = current_player();
  for (int i = 0; i < kNumOfGrids; ++i) {
    tensor.push_back(static_cast<float>(board_[i] == player));
  }
  for (int i = 0; i < kNumOfGrids; ++i) {
    tensor.push_back(static_cast<float>(board_[i] == 1 - player));
  }
  for (int i = 0; i < kNumOfGrids; ++i) {
    tensor.push_back(static_cast<float>((board_[i] == player) && kIsInter[i]));
  }
  for (int i = 0; i < kNumOfGrids; ++i) {
    tensor.push_back(static_cast<float>((board_[i] == 1 - player) && kIsInter[i]));
  }
  for (int i = 0; i < kNumOfGrids; ++i) {
    tensor.push_back(static_cast<float>((board_[i] == player) && kIsOuter[i]));
  }
  for (int i = 0; i < kNumOfGrids; ++i) {
    tensor.push_back(static_cast<float>((board_[i] == 1 - player) && kIsOuter[i]));
  }
  for (int i = 0; i < kNumOfGrids; ++i) {
    tensor.push_back(
        static_cast<float>((turn_ % 2) && (board_[i] == history_[history_.size() - 1])));
  }
  for (int i = 0; i < kNumOfGrids; ++i) {
    tensor.push_back(static_cast<float>((turn_ % 2)));
  }
  return tensor;
}

bool SurakartaSplitActionState::has_piece(const Player &player) const {
  return (player == BLACK && black_piece_ != 0) || (player == WHITE && white_piece_ != 0);
}

std::vector<float> SurakartaSplitActionState::rewards() const {
  if (winner_ == BLACK) {
    return {1.0, -1.0};
  }
  if (winner_ == WHITE) {
    return {-1.0, 1.0};
  }
  if (legal_actions().empty()) {
    if (black_piece_ > white_piece_) {
      return {1.0, -1.0};
    }
    if (black_piece_ < white_piece_) {
      return {-1.0, 1.0};
    }
  }
  return {0.0, 0.0};
}

std::string SurakartaSplitActionState::serialize() const {
  std::stringstream ss;
  if (history_.size() % 2 == 1) {
    ss << -1 << " ";
    for (const Action action : history_) {
      ss << action << " ";
    }
  } else {
    for (int i = 0; i < history_.size(); i += 2) {
      Action action = history_[i] * kNumOfGrids + history_[i + 1];
      ss << action << " ";
    }
  }
  return ss.str();
}

std::string SurakartaSplitActionGame::name() const { return "surakarta_split_action"; }
int SurakartaSplitActionGame::num_players() const { return 2; }
int SurakartaSplitActionGame::num_distinct_actions() const { return kPolicyDim; }
std::vector<int> SurakartaSplitActionGame::observation_tensor_shape() const {
  return {8, kBoardSize, kBoardSize};
}

StatePtr SurakartaSplitActionGame::new_initial_state() const {
  return std::make_unique<SurakartaSplitActionState>(shared_from_this());
}

int SurakartaSplitActionGame::num_transformations() const { return 2; }

std::vector<float> SurakartaSplitActionGame::transform_observation(
    const std::vector<float> &observation, int type) const {
  std::vector<float> data(observation);
  if (type != 0) {
    int feature_num = (int)observation.size() / kNumOfGrids;
    for (int i = 0; i < feature_num; i++) {
      for (int index = 0; index < kNumOfGrids; index++) {
        int new_index = transform_index(index, type);
        data[i * kNumOfGrids + new_index] = observation[i * kNumOfGrids + index];
      }
    }
  }
  return data;
}

std::vector<float> SurakartaSplitActionGame::transform_policy(const std::vector<float> &policy,
                                                              int type) const {
  std::vector<float> data(policy);
  if (type != 0) {
    for (int index = 0; index < kPolicyDim; index++) {
      int new_index = transform_index(index, type);
      data[new_index] = policy[index];
    }
  }
  return data;
}

std::vector<float> SurakartaSplitActionGame::restore_policy(const std::vector<float> &policy,
                                                            int type) const {
  std::vector<float> data(policy);
  if (type != 0) {
    for (int index = 0; index < kPolicyDim; index++) {
      int new_index = transform_index(index, type);
      data[new_index] = policy[index];
    }
  }
  return data;
}

int SurakartaSplitActionGame::transform_index(const int &index, const int type) const {
  if (type == 0) {
    return index;
  } else {
    int i = index / kBoardSize;
    int j = index % kBoardSize;
    if (type == 1) {
      j = (kBoardSize - 1) - j;
    }
    return i * kBoardSize + j;
  }
}

std::string SurakartaSplitActionGame::action_to_string(const Action &action) const {
  using namespace std::string_literals;
  std::stringstream ss;
  ss << "ABCDEF"s.at(action % kBoardSize) << kBoardSize - (action / kBoardSize);
  return ss.str();
}

std::vector<Action> SurakartaSplitActionGame::string_to_action(const std::string &str) const {
  std::string wopunc = str;
  wopunc.erase(std::remove_if(wopunc.begin(), wopunc.end(), ispunct), wopunc.end());
  std::stringstream ss(wopunc);
  std::string action;
  std::vector<Action> id;
  while (ss >> action) {
    if (action.at(0) >= 'a' && action.at(0) <= 'z') {
      int tmp = (kBoardSize - std::stoi(action.substr(1))) * kBoardSize + (action.at(0) - 'a');
      id.push_back(tmp);
    } else {
      int tmp = (kBoardSize - std::stoi(action.substr(1))) * kBoardSize + (action.at(0) - 'A');
      id.push_back(tmp);
    }
  }
  return id;
}

StatePtr SurakartaSplitActionGame::deserialize_state(const std::string &str) const {
  std::stringstream ss(str);
  bool mode = false;
  int action;
  StatePtr state = new_initial_state();
  while (ss >> action) {
    if (action == -1) {
      mode = true;
      continue;
    }
    if (mode) {
      state->apply_action(action);
    } else {
      int src = action / 36;
      int des = action % 36;
      state->apply_action(src);
      state->apply_action(des);
    }
  }
  return state->clone();
}

}  // namespace czf::env::czf_env::surakarta