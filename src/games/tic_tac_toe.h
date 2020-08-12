#ifndef CZF_GAMES_TIC_TAC_TOE_H_
#define CZF_GAMES_TIC_TAC_TOE_H_

#include <array>

#include "czf.h"

namespace czf {
namespace tic_tac_toe {

class TicTacToeState : public State {
 public:
  TicTacToeState(std::shared_ptr<const Game> game);
  TicTacToeState(std::shared_ptr<const Game> game, const std::string &);
  void ApplyAction(const Action &) override;
  std::vector<Action> LegalActions() const override;
  bool IsTerminal() const override;
  Player CurrentPlayer() const override;
  std::vector<float> ObservationTensor() const override;
  std::string Serialize() const override;

  bool HasLine(const Player &player) const;

 private:
  // 0 -> O
  // 1 -> X
  // 2 -> empty
  std::array<short, 9> board_;
  int turn_;
  Player winner_;
};

class TicTacToeGame : public Game {
 public:
  std::string ShortName() const override;
  std::string Name() const override;
  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override;
  std::unique_ptr<State> DeserializeState(const std::string &) const override;
  std::vector<int> ObservationTensorShape() const override;
};

REGISTER_CZF_GAME(TicTacToeGame)

}  // namespace tic_tac_toe
}  // namespace czf

#endif  // CZF_GAMES_TIC_TAC_TOE_H_