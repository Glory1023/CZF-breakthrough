#include "games/tic_tac_toe.h"

namespace czf {
namespace tic_tac_toe {
TicTacToeState::TicTacToeState(std::shared_ptr<const Game> game)
    : State(game), turn_(0), winner_(-1) {
  board_.fill(2);
}

TicTacToeState::TicTacToeState(std::shared_ptr<const Game> game,
                               const std::string &state)
    : State(game), winner_(-1) {
  int empty = 0;
  for (int i = 0; i < 9; ++i) {
    switch (state[i]) {
      case 'O':
        board_[i] = 0;
        break;
      case 'X':
        board_[i] = 1;
        break;
      default:
        board_[i] = 2;
        ++empty;
    }
  }
  turn_ = 9 - empty;
  if (HasLine(0))
    winner_ = 0;
  else if (HasLine(1))
    winner_ = 1;
}

void TicTacToeState::ApplyAction(const Action &action) {
  Player player = CurrentPlayer();
  board_[action] = player;
  ++turn_;
  if (HasLine(player)) winner_ = player;
}

std::vector<Action> TicTacToeState::LegalActions() const {
  std::vector<Action> actions;
  actions.reserve(9 - turn_);
  for (int i = 0; i < 9; ++i)
    if (board_[i] == 2) actions.push_back(i);
  return actions;
}

bool TicTacToeState::IsTerminal() const { return turn_ == 9 || winner_ != -1; }

Player TicTacToeState::CurrentPlayer() const { return turn_ % 2; }

std::vector<float> TicTacToeState::ObservationTensor() const {
  std::vector<float> tensor;
  tensor.reserve(27);
  Player player = CurrentPlayer();
  for (int i = 0; i < 9; ++i) tensor.push_back(board_[i] == player);
  for (int i = 0; i < 9; ++i) tensor.push_back(board_[i] == 1 - player);
  for (int i = 0; i < 9; ++i) tensor.push_back(player);
  return tensor;
}

std::string TicTacToeState::Serialize() const {
  std::string serialized;
  serialized.reserve(9);
  for (int i = 0; i < 9; ++i) serialized += "OX "[board_[i]];
  return serialized;
}

bool TicTacToeState::HasLine(const Player &player) const {
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

std::string TicTacToeGame::ShortName() const { return "tic_tac_toe"; }
std::string TicTacToeGame::Name() const { return "Tic Tac Toe"; }
int TicTacToeGame::NumDistinctActions() const { return 9; }
std::vector<int> TicTacToeGame::ObservationTensorShape() const {
  return {3, 3, 3};
}

std::unique_ptr<State> TicTacToeGame::NewInitialState() const {
  return std::unique_ptr<State>(new TicTacToeState(shared_from_this()));
}

std::unique_ptr<State> TicTacToeGame::DeserializeState(
    const std::string &state) const {
  return std::unique_ptr<State>(new TicTacToeState(shared_from_this(), state));
}

}  // namespace tic_tac_toe
}  // namespace czf