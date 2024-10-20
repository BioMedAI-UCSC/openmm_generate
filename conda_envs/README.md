

# Running on delta

Conda uses a "virtual package" `__cuda` which is based off the installed version on the system, but the login nodes don't have Cuda installed so you have to do something like this:

```
CONDA_OVERRIDE_CUDA=12.2 conda env create --file=./delta.yml --name <BLAH BLAH>
```

