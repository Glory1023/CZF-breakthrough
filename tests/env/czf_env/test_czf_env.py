import unittest
from czf.env import czf_env


class czf_env_Test(unittest.TestCase):
    def setUp(self):
        self.game_name = 'tic_tac_toe'

    def test_available_games(self):
        self.assertTrue(len(czf_env.available_games()) > 0)

    def test_game_name(self):
        game = czf_env.load_game(self.game_name)
        self.assertEqual(self.game_name, game.name)

    def test_game_shape(self):
        game = czf_env.load_game(self.game_name)
        self.assertEqual(game.observation_tensor_shape, [4, 3, 3])

    def test_new_initial_state(self):
        game = czf_env.load_game(self.game_name)
        state = game.new_initial_state()
        self.assertEqual(state.current_player, 0)
        self.assertFalse(state.is_terminal)

    def test_steps(self):
        game = czf_env.load_game(self.game_name)
        state = game.new_initial_state()
        for i in range(3):
            self.assertEqual(state.current_player, i % 2)
            action = state.legal_actions[-1]
            state.apply_action(action)
            self.assertFalse(state.is_terminal)


if __name__ == '__main__':
    unittest.main()