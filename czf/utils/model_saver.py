'''Model Saver'''
import argparse
from io import BytesIO
import os
from pathlib import Path

import torch
import torch.jit
import zstandard as zstd


def jit_alphazero(buffer, device='cuda'):
    '''jit model'''
    state_dict = torch.load(BytesIO(buffer))
    model = state_dict['model'].to(device)
    iteration = state_dict['iteration']
    observation_shape = state_dict['observation_shape']
    model._is_train = False
    model.eval()
    # JIT
    buffer = BytesIO()
    input_obs = torch.rand(1, *observation_shape).to(device)
    with torch.jit.optimized_execution(True):
        frozen_net = torch.jit.trace_module(
            model, {
                'forward': (input_obs, ),
            })
        torch.jit.save(frozen_net, buffer)
    buffer.seek(0)
    return iteration, buffer.read()


def jit_muzero(buffer, device='cuda'):
    '''jit model'''
    state_dict = torch.load(BytesIO(buffer))
    model = state_dict['model'].to(device)
    iteration = state_dict['iteration']
    observation_shape = state_dict['observation_shape']
    state_shape = state_dict['state_shape']
    model._is_train = False
    model.eval()
    # JIT
    buffer = BytesIO()
    input_obs = torch.rand(1, *observation_shape).to(device)
    input_state = torch.rand(1, *state_shape).to(device)
    input_action = torch.rand(1, 1).to(device)
    with torch.jit.optimized_execution(True):
        frozen_net = torch.jit.trace_module(
            model, {
                'forward_representation': (input_obs, ),
                'forward_dynamics': (input_state, input_action),
                'forward': (input_state, ),
            })
        torch.jit.save(frozen_net, buffer)
    buffer.seek(0)
    return iteration, buffer.read()


def save(args, iteration, buffer):
    '''save jit model'''
    model_dir = Path(args.model_dir)
    model_path = model_dir / f'{iteration:05d}.pt.zst'
    cctx = zstd.ZstdCompressor()
    buffer = cctx.compress(buffer)
    model_path.write_bytes(buffer)
    # update the latest model file
    latest_model = model_dir / 'latest.pt.zst'
    temp_model = model_dir / 'latest-temp.pt.zst'
    os.symlink(model_path, temp_model)
    os.replace(temp_model, latest_model)
    if args.rm:
        os.remove(args.checkpoint)


def main(args):
    '''czf.utils.model_server main program'''
    device = 'cuda'
    # load checkpoint
    with open(args.checkpoint, 'rb') as model_blob:
        buffer = model_blob.read()
        # dctx = zstd.ZstdDecompressor()
        # buffer = dctx.decompress(buffer)
    if args.algorithm == 'AlphaZero':
        iteration, buffer = jit_alphazero(buffer, device)
    elif args.algorithm == 'MuZero':
        iteration, buffer = jit_muzero(buffer, device)
    save(args, iteration, buffer)


def run_main():
    '''Run main program'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',
                        help='path to load checkpoint')
    parser.add_argument('--model-dir',
                        help='directory to save model')
    parser.add_argument('--algorithm',
                        help='used algorithm (AlphaZero or MuZero)')
    parser.add_argument('--rm',
                        action='store_true',
                        help='remove the checkpoint')
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    run_main()
