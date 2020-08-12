#ifndef CZF_CZF_H_
#define CZF_CZF_H_

#define REGISTER_CZF_GAME(CZF_GAME)                     \
  namespace {                                           \
  GameRegistration game_registration([]() {             \
    return std::shared_ptr<const Game>(new CZF_GAME()); \
  });                                                   \
  }

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace czf {

using Player = int;
using Action = int;

class Game;

class State {
 public:
  State(std::shared_ptr<const Game> game) : game_(game){};
  virtual void ApplyAction(const Action &) = 0;
  virtual std::vector<Action> LegalActions() const = 0;
  virtual bool IsTerminal() const = 0;
  virtual Player CurrentPlayer() const = 0;
  virtual std::vector<float> ObservationTensor() const = 0;
  virtual std::string Serialize() const = 0;
  std::shared_ptr<const Game> GetGame() const { return game_; };

 protected:
  std::shared_ptr<const Game> game_;
};

class Game : public std::enable_shared_from_this<Game> {
 public:
  virtual ~Game() = default;
  virtual std::string ShortName() const = 0;
  virtual std::string Name() const = 0;
  virtual int NumDistinctActions() const = 0;
  virtual std::unique_ptr<State> NewInitialState() const = 0;
  virtual std::unique_ptr<State> DeserializeState(
      const std::string &) const = 0;
  // Channel, Height, Width
  virtual std::vector<int> ObservationTensorShape() const = 0;
  virtual std::vector<int> PolicyTensorShape() const {
    return {NumDistinctActions()};
  }
};

class GameRegistration {
  using CreateFunc = std::function<std::shared_ptr<const Game>()>;

 public:
  GameRegistration(CreateFunc);
  static std::shared_ptr<const Game> Create(const std::string &);

 private:
  static std::map<std::string, CreateFunc> &factories();
};

std::shared_ptr<const Game> LoadGame(const std::string &short_name);

}  // namespace czf

#endif  // CZF_CZF_H_