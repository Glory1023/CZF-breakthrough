# czf: CGI Zero Framework

## Install

```shell
pip install .
```

## Examples

### Run Locally

- Create a training configuration: `game.yaml` (ref: `config.example.yaml`)
- Run the following commands in different shells:

```shell
czf-actor -b 127.0.0.1:5566 -u 127.0.0.1:5577 -f game.yaml
czf-learner -l 5577 -f game.yaml
czf-broker -l 5566
czf-game-server -b 127.0.0.1:5566 -u 127.0.0.1:5577 -g tic_tac_toe -n 1
```

## Development

```shell
python setup.py build -g
```

- Format: `python setup.py format`
- Test: `python setup.py tests`

### Protobuf

- protoc 3.12.4
- Compile: `python setup.py proto`
