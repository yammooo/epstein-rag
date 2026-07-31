# NLP project

Install the common dependencies:

```bash
python -m pip install -r requirements.txt
```

`llama-cpp-python` is hardware-specific and is installed separately. For CPU:

```bash
python -m pip install llama-cpp-python
```

For CUDA 12.4:

```bash
python -m pip install --upgrade --force-reinstall 
  llama-cpp-python 
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```
