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
usage: [-h] [-o [OUTPUT]] [-v] platform pip_argument [pip_argument ...]

positional arguments:
  platform
  pip_argument

optional arguments:
  -h, --help            show this help message and exit
  -o [OUTPUT], --output [OUTPUT]
  -v, --verbose
```
