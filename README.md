# czf: CGI Zero Framework
For more documentation, please refer to [here](https://hackmd.io/@CGI-Lab/HJsi2hrV_)

## Get Repository
- Clone both repository and submodule
```bash
git clone --recursive https://github.com/kaocy/czf
```

- If the submodule is empty after `git clone`, type the following command
```bash
git submodule update --init --recursive
```

## Podman usage
- Download the image
```shell
podman pull chengscott/czf:devel
```

- Run container
```shell
podman run --rm -d -v $PWD:/czf -w /czf --name="czf" -it --network=host --shm-size 8g chengscott/czf:devel
```

- Create a bash shell in the container (for installation and training, create a shell in the container before running commands)
```shell
podman exec -w=/czf -it czf bash
```

## Install (TODO: Build a new image for missing packages)
```shell
pip install psutil

# Either release build
python setup.py build

# Or debug build
python setup.py build -g
```

## Run
- Create a training configuration: `game.yaml` (ref: `config/muzero/tic_tac_toe.yaml`)
- For each shell, you need to enter the build directory (e.g., `build/lib.linux-x86_64-3.8`)
- Run the following commands in different shells:

### Training

```bash=
python -m czf.learner -l 5577 -f game.yaml
python -m czf.broker -l 5566
python -m czf.actor -b 127.0.0.1:5566 -u 127.0.0.1:5577 -f game.yaml -bs 64
python -m czf.game_server -b 127.0.0.1:5566 -u 127.0.0.1:5577 -f game.yaml -np 1 -n 128
```

### CLI Agent

- Currently only support MuZero

```bash=
python -m czf.model_provider -l 5577 -s $storage_dir
python -m czf.broker -l 5566
python -m czf.actor -b 127.0.0.1:5566 -u 127.0.0.1:5577 -f game.yaml -bs 1
python -m czf.cli_agent -b 127.0.0.1:5566 -f game.yaml -v $model_iteration
```
