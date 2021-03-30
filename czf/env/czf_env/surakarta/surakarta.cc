#include "czf/env/czf_env/surakarta/surakarta.h"

#include <algorithm>
#include <iomanip>
#include <numeric>
#include <sstream>

namespace czf::env::czf_env::surakarta {

SurakartaState::SurakartaState(GamePtr game_ptr)
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

StatePtr SurakartaState::clone() const {
  return std::make_unique<SurakartaState>(*this);
}

void SurakartaState::apply_action(const Action &action) {
  Player player = current_player();
  const int src = action / kNumOfGrids;
  const int des = action % kNumOfGrids;
  if (board_[des] != EMPTY) {
    without_capture_turn_ = 0;
    repeat_.clear();
    if (board_[des] == BLACK) {
      black_piece_--;
    } else if (board_[des] == WHITE) {
      white_piece_--;
    }
  } else {
    without_capture_turn_++;
  }
  std::swap(board_[src], board_[des]);
  board_[src] = EMPTY;
  ++turn_;
  history_.push_back(action);
  repeat_[board_]++;

  if (!has_piece(1 - player)) {
    winner_ = player;
  } else if (repeat_[board_] >= kMaxRepeatBoard ||
             without_capture_turn_ >= kMaxNoCaptureTurn || turn_ >= kMaxTurn) {
    if (black_piece_ > white_piece_) {
      winner_ = BLACK;
    } else if (black_piece_ < white_piece_) {
      winner_ = WHITE;
    } else if (black_piece_ == white_piece_) {
      winner_ = EMPTY;
    }
  }
}

std::vector<Action> SurakartaState::legal_actions() const {
  std::vector<Action> actions;
  std::vector<bool> Bool = get_is_legal_action();
  for (uint32_t i = 0; i < kPolicyDim; i++) {
    if (Bool[i]) {
      actions.push_back(i);
    }
  }
  return actions;
}

void SurakartaState::get_is_legal_capture(
    std::vector<bool> &is_legal, const std::array<int, 56> &circle) const {
  int our_index = -1;
  bool CanCapture = false;
  for (int src_index = 0; src_index < 56; src_index++) {
    if (circle[src_index] == -1) {
      CanCapture = true;
    } else if (current_player() == board_[circle[src_index]]) {
      if (our_index < 0 || circle[src_index] != circle[our_index]) {
        our_index = src_index;
        CanCapture = false;
      }
    } else if (1 - current_player() == board_[circle[src_index]]) {
      if (CanCapture && our_index >= 0) {
        uint32_t id = kNumOfGrids * circle[our_index] + circle[src_index];
        is_legal[id] = true;
      }
      our_index = -1;
      CanCapture = false;
    }
    if (our_index >= 28 || (our_index < 0 && src_index >= 28)) {
      break;
    }
  }
}

std::vector<bool> SurakartaState::get_is_legal_action() const {
  std::vector<bool> is_legal(kPolicyDim, false);
  // move
  for (int src_y = 0; src_y < kBoardSize; src_y++) {
    for (int src_x = 0; src_x < kBoardSize; src_x++) {
      uint32_t src = src_y * kBoardSize + src_x;
      if (current_player() == board_[src]) {
        for (int des_y = src_y - 1; des_y <= src_y + 1; des_y++) {
          for (int des_x = src_x - 1; des_x <= src_x + 1; des_x++) {
            if (des_y >= 0 && des_y < kBoardSize && des_x >= 0 &&
                des_x < kBoardSize) {
              uint32_t des = des_y * kBoardSize + des_x;
              if (board_[des] == EMPTY) {
                uint32_t id = kNumOfGrids * src + des;
                is_legal[id] = true;
              }
            }
          }
        }
      }
    }
  }
  // capture
  get_is_legal_capture(is_legal, kOuter);
  get_is_legal_capture(is_legal, kOuterReverse);
  get_is_legal_capture(is_legal, kInter);
  get_is_legal_capture(is_legal, kInterReverse);

  return is_legal;
}

bool SurakartaState::no_leagal_actions() const {
  for (int src_y = 0; src_y < kBoardSize; src_y++) {
    for (int src_x = 0; src_x < kBoardSize; src_x++) {
      uint32_t src = src_y * kBoardSize + src_x;
      if (current_player() == board_[src]) {
        for (int des_y = src_y - 1; des_y <= src_y + 1; des_y++) {
          for (int des_x = src_x - 1; des_x <= src_x + 1; des_x++) {
            if (des_y >= 0 && des_y < kBoardSize && des_x >= 0 &&
                des_x < kBoardSize) {
              uint32_t des = des_y * kBoardSize + des_x;
              if (board_[des] == EMPTY) {
                return false;
              }
            }
          }
        }
      }
    }
  }
  if (!no_legal_capture(kOuter)) return false;
  if (!no_legal_capture(kOuterReverse)) return false;
  if (!no_legal_capture(kInter)) return false;
  if (!no_legal_capture(kInterReverse)) return false;
  return true;
}
bool SurakartaState::no_legal_capture(const std::array<int, 56> &circle) const {
  int our_index = -1;
  bool CanCapture = false;
  for (int src_index = 0; src_index < 56; src_index++) {
    if (circle[src_index] == -1) {
      CanCapture = true;
    } else if (current_player() == board_[circle[src_index]]) {
      if (our_index < 0 || circle[src_index] != circle[our_index]) {
        our_index = src_index;
        CanCapture = false;
      }
    } else if (1 - current_player() == board_[circle[src_index]]) {
      if (CanCapture && our_index >= 0) {
        return false;
      }
      our_index = -1;
      CanCapture = false;
    }
    if (our_index >= 28 || (our_index < 0 && src_index >= 28)) {
      break;
    }
  }
  return true;
}

std::string SurakartaState::to_string() const {
  std::stringstream ss;
  const std::vector<std::string> chess{"B", "W", "·"};

  for (int i = 0; i < kBoardSize; ++i) {
    ss << std::setw(2) << std::setfill(' ') << kBoardSize - i;
    for (int j = 0; j < kBoardSize; ++j)
      ss << ' ' << chess[board_[i * kBoardSize + j]];
    ss << std::endl;
  }
  ss << "  ";
  for (int i = 0; i < kBoardSize; ++i) ss << ' ' << static_cast<char>('A' + i);
  return ss.str();
}

bool SurakartaState::is_terminal() const {
  return (winner_ != -1) || (no_leagal_actions());
}

Player SurakartaState::current_player() const { return turn_ % 2; }

std::vector<float> SurakartaState::observation_tensor() const {
  std::vector<float> tensor;
  tensor.reserve(6 * 6 * 6);
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
    tensor.push_back(
        static_cast<float>((board_[i] == 1 - player) && kIsInter[i]));
  }
  for (int i = 0; i < kNumOfGrids; ++i) {
    tensor.push_back(static_cast<float>((board_[i] == player) && kIsOuter[i]));
  }
  for (int i = 0; i < kNumOfGrids; ++i) {
    tensor.push_back(
        static_cast<float>((board_[i] == 1 - player) && kIsOuter[i]));
  }
  return tensor;
}

bool SurakartaState::has_piece(const Player &player) const {
  return (player == BLACK && black_piece_ != 0) ||
         (player == WHITE && white_piece_ != 0);
}

std::vector<float> SurakartaState::rewards() const {
  if (winner_ == BLACK) {
    return {1.0, -1.0};
  }
  if (winner_ == WHITE) {
    return {-1.0, 1.0};
  }
  if (no_leagal_actions()) {
    if (black_piece_ > white_piece_) {
      return {1.0, -1.0};
    }
    if (black_piece_ < white_piece_) {
      return {-1.0, 1.0};
    }
  }
  return {0.0, 0.0};
}

std::string SurakartaState::serialize() const {
  std::stringstream ss;
  for (const Action action : history_) {
    ss << action << " ";
  }
  return ss.str();
}

std::string SurakartaGame::name() const { return "surakarta"; }
int SurakartaGame::num_players() const { return 2; }
int SurakartaGame::num_distinct_actions() const { return kPolicyDim; }
std::vector<int> SurakartaGame::observation_tensor_shape() const {
  return {6, kBoardSize, kBoardSize};
}

StatePtr SurakartaGame::new_initial_state() const {
  return std::make_unique<SurakartaState>(shared_from_this());
}

int SurakartaGame::num_transformations() const { return 4; }

std::vector<float> SurakartaGame::transform_observation(
    const std::vector<float> &observation, int type) const {
  std::vector<float> data(observation);
  if (type != 0) {
    int feature_num = (int)observation.size() / kNumOfGrids;
    for (int i = 0; i < feature_num; i++) {
      for (int index = 0; index < kNumOfGrids; index++) {
        int new_index = transform_index(index, type);
        data[i * kNumOfGrids + new_index] =
            observation[i * kNumOfGrids + index];
      }
    }
  }
  return data;
}

std::vector<float> SurakartaGame::transform_policy(
    const std::vector<float> &policy, int type) const {
  std::vector<float> data(policy);
  if (type != 0) {
    for (int index = 0; index < kPolicyDim; index++) {
      int sur_index = transform_index(index / 36, type);
      int des_index = transform_index(index % 36, type);
      int new_index = sur_index * 36 + des_index;
      data[new_index] = policy[index];
    }
  }
  return data;
}

std::vector<float> SurakartaGame::restore_policy(
    const std::vector<float> &policy, int type) const {
  std::vector<float> data(policy);
  if (type != 0) {
    for (int index = 0; index < kPolicyDim; index++) {
      int sur_index = transform_index(index / 36, type);
      int des_index = transform_index(index % 36, type);
      int new_index = sur_index * 36 + des_index;
      data[new_index] = policy[index];
    }
  }
  return data;
}

int SurakartaGame::transform_index(const int &index, const int type) const {
  if (type == 0) {
    return index;
  } else {
    int i = index / kBoardSize;
    int j = index % kBoardSize;
    if (type == 1) {
      i = (kBoardSize - 1) - i;
      j = (kBoardSize - 1) - j;
    } else if (type == 2) {
      i = (kBoardSize - 1) - i;
    } else if (type == 3) {
      j = (kBoardSize - 1) - j;
    }
    return i * kBoardSize + j;
  }
}

std::string SurakartaGame::action_to_string(const Action &action) const {
  using namespace std::string_literals;
  std::stringstream ss;
  int src = action / kNumOfGrids;
  int des = action % kNumOfGrids;
  ss << "ABCDEF"s.at(src % kBoardSize) << kBoardSize - (src / kBoardSize)
     << ", "
     << "ABCDEF"s.at(des % kBoardSize) << kBoardSize - (des / kBoardSize);
  return ss.str();
}

std::vector<Action> SurakartaGame::string_to_action(
    const std::string &str) const {
  std::string wopunc = str;
  wopunc.erase(std::remove_if(wopunc.begin(), wopunc.end(), ispunct),
               wopunc.end());
  std::stringstream ss(wopunc);
  std::string action;
  std::vector<Action> id;
  int tmp = 0;
  while (ss >> action) {
    tmp *= kNumOfGrids;
    if (action.at(0) >= 'a' && action.at(0) <= 'z') {
      tmp += (kBoardSize - std::stoi(action.substr(1))) * kBoardSize +
             (action.at(0) - 'a');
    } else {
      tmp += (kBoardSize - std::stoi(action.substr(1))) * kBoardSize +
             (action.at(0) - 'A');
    }
  }
  id.push_back(tmp);
  return id;
}

StatePtr SurakartaGame::deserialize_state(const std::string &str) const {
  std::stringstream ss(str);
  int action;
  StatePtr state = new_initial_state();
  while (ss >> action) {
    state->apply_action(action);
  }
  return state->clone();
}

}  // namespace czf::env::czf_env::surakarta