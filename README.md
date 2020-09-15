# czf: CGI Zero Framework

## Install

```shell
pip install ./czf
```

## Examples

### Run Locally

- Run the following commands in three shells:

```shell
czf-actor -b 127.0.0.1:5566 -g tic_tac_toe
czf-broker -l 5566
czf-game-server -b 127.0.0.1:5566 -g tic_tac_toe -n 1
```

## Development

- protoc 3.12.4
