'''Model Saver'''
import argparse
from io import BytesIO
import os
from pathlib import Path

import torch
import torch.jit
import zstandard as zstd


def main(args):
    '''czf.utils.model_server main program'''
    device = 'cuda'
    # load checkpoint
    with open(args.checkpoint, 'rb') as model_blob:
        dctx = zstd.ZstdDecompressor()
        ckpt = dctx.decompress(model_blob.read())
    state_dict = torch.load(BytesIO(ckpt))
    model = state_dict['model'].to(device)
    iteration = state_dict['iteration']
    observation_shape = state_dict['observation_shape']
    state_shape = state_dict['state_shape']
    model.is_train = False
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
    # save model
    model_path = Path(args.model_path)
    buffer.seek(0)
    cctx = zstd.ZstdCompressor()
    compressed = cctx.compress(buffer.read())
    model_file = model_path / f'{iteration:05d}.pt.zst'
    model_file.write_bytes(compressed)
    # update the latest model file
    latest_model = model_path / 'latest.pt.zst'
    temp_model = model_path / 'latest-temp.pt.zst'
    os.symlink(model_file, temp_model)
    os.replace(temp_model, latest_model)
    if args.rm:
        os.remove(args.checkpoint)


def run_main():
    '''Run main program'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',
                        required=True,
                        help='path to load checkpoint')
    parser.add_argument('--model-path',
                        required=True,
                        help='path to save model')
    parser.add_argument('--rm',
                        action='store_true',
                        help='remove the checkpoint')
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    run_main()
