#include <iostream>
#include <random>

#include "third_party/czf_env/czf_env/czf_env.h"
using namespace std;

int main() {
  std::random_device rd;
  std::mt19937 gen(rd());

  auto game = czf_env::LoadGame("tic_tac_toe");

  auto state = game->NewInitialState();
  while (!state->IsTerminal()) {
    auto legal_actions = state->LegalActions();
    uniform_int_distribution<int> distribution(0, legal_actions.size() - 1);
    auto action_index = distribution(gen);
    state->ApplyAction(legal_actions[action_index]);
    cout << state->Serialize() << endl;
  }
  return 0;
}