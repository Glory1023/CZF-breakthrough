#! /usr/bin/env python3
import random
import pyczf


def main():
    game = pyczf.load_game("tic_tac_toe")
    state = game.new_initial_state()
    while not state.is_terminal():
        legal_actions = state.legal_actions()
        action = random.choice(legal_actions)
        state.apply_action(action)
        print(state.serialize())


if __name__ == "__main__":
    main()
