from torch.utils.data import Dataset


class ReplayBuffer(Dataset):
    def __init__(self, game, capacity, train_freq):
        self.game = game
        self.capacity = capacity
        self.train_freq = train_freq

        self.data = []
        self.num_new_states = 0
        self.ready = False

        self.count = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def add_trajectory(self, trajectory):
        self.count += 1
        if self.count % 100 == 0 and trajectory.returns[0]:
            game = czf.game.load('tic_tac_toe')
            game_state = game.new_initial_state()
            for state in trajectory.states:
                clone_policy = [p for p in state.evaluation.policy]
                clone_policy.sort()
                for p in state.evaluation.policy:
                    print(f'{clone_policy.index(p):5d}', end=' ')
                print()
                for p in state.evaluation.policy:
                    print(f'{p:.3f}', end=' ')
                print()
                game_state.apply_action(state.transition.action)
                print(game_state)

        print(' '.join(f'{p:.3f}'
                       for p in trajectory.states[0].evaluation.policy))
        print(trajectory.returns)

        for state in trajectory.states:
            # print(game_state)
            # clone_policy = [p for p in state.evaluation.policy]
            # clone_policy.sort()
            # for p in state.evaluation.policy:
            #     print(f'{clone_policy.index(p):5d}', end=' ')
            # print()
            # for p in state.evaluation.policy:
            #     print(f'{p:.3f}', end=' ')
            # print()
            # game_state.apply_action(state.transition.action)

            # self.total += 1
            # print(self.total)
            self.num_new_states += 1

            observation_tensor = torch.tensor(state.observation_tensor).view(
                self.game.observation_tensor_shape)
            policy = torch.tensor(state.evaluation.policy)
            value = torch.tensor(trajectory.returns)

            self.data.append((observation_tensor, policy, value))
            if len(self.data) > self.capacity:
                self.data.pop(0)
        # print(game_state)

        # print(trajectory.returns)

        if self.num_new_states >= self.train_freq:
            self.ready = True
            self.num_new_states -= self.train_freq