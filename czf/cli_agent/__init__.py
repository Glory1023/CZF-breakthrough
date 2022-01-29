'''CZF CLI Agent'''
import argparse
import asyncio
from functools import partial
from pathlib import Path
from uuid import uuid4

import numpy as np
import yaml
import zmq.asyncio

from czf.cli_agent.cli_agent import CliAgent


def argmax_action_policy_fn(_, legal_actions, legal_actions_policy):
    '''Argmax action policy'''
    return legal_actions[np.argmax(legal_actions_policy)]


def eval_after_apply_metric(board_shape, evaluated_state):
    '''Callback for the evaluation result after applying an action'''
    policy = evaluated_state.evaluation.policy
    value = evaluated_state.evaluation.value
    print(np.array(policy).reshape(board_shape), value, sep='\n')


async def main(args, config, callbacks):
    '''czf.game_server main program'''
    agent = CliAgent(args, config, callbacks)
    await agent.send_load_model()
    await agent.loop()


def run_main():
    '''Run main program in asyncio'''
    parser = argparse.ArgumentParser(__package__, description=__doc__)
    parser.add_argument('-f', '--config', required=True, help='config file')
    parser.add_argument('-b',
                        '--broker',
                        required=True,
                        metavar='host:port',
                        help='broker address. e.g., 127.0.0.1:5566')
    parser.add_argument('--suffix',
                        metavar='unique_id',
                        default=uuid4().hex,
                        help='unique id of the game server')
    parser.add_argument('-v', '--model-version', type=int, default=-1, help='model version')
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    softmax_step = config['game_server']['softmax_temperature_step']
    callbacks = {
        'metric': {},
    }
    callbacks['action_policy'] = argmax_action_policy_fn
    np.set_printoptions(precision=3)
    board_shape = config['game']['observation']['spatial_shape'][-2:]
    callbacks['metric']['after_apply'] = partial(eval_after_apply_metric, board_shape)
    try:
        asyncio.run(main(args, config, callbacks))
    except KeyboardInterrupt:
        zmq.asyncio.Context.instance().destroy()
        print('\rterminated by ctrl-c')


if __name__ == '__main__':
    run_main()
