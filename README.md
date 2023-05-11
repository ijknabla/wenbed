# wenbed
python embed for windows generator

# run

```
curl -sSL https://raw.githubusercontent.com/ijknabla/wenbed/main/wenbed.py \
    | python - ${OPTIONS} ${ARGUMENTS}
```

```
wget https://raw.githubusercontent.com/ijknabla/wenbed/main/wenbed.py -q -O- \
    | python - ${OPTIONS} ${ARGUMENTS}
```

```
(Invoke-WebRequest -Uri https://raw.githubusercontent.com/ijknabla/wenbed/main/wenbed.py -UseBasicParsing).Content `
    | python - ${OPTIONS} ${ARGUMENTS}
```

# Options & Arguments

```
usage: wenbed [-h] [-o [OUTPUT]] [-v] platform [pip_argument ...]

positional arguments:
  platform
  pip_argument

options:
  -h, --help            show this help message and exit
  -o [OUTPUT], --output [OUTPUT]
  -v, --verbose
```

# Example

## Install numpy into python 3.9.0 (64-bit) & 3.10.0 (64-bit)

```
(Invoke-WebRequest -Uri https://raw.githubusercontent.com/ijknabla/wenbed/main/wenbed.py -UseBasicParsing).Content `
    | python - 3.9.0-amd64,3.10.0-amd64 -- install numpy
```
