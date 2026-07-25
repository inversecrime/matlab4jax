### Matlab library path

On Linux, set the Matlab library path before importing, for example:
```bash
export LD_LIBRARY_PATH=/usr/local/MATLAB/R2026a/extern/bin/glnxa64
```

To make this setting persistent, add it to your `~/.bashrc`:
```bash
echo 'export LD_LIBRARY_PATH=/usr/local/MATLAB/R2026a/extern/bin/glnxa64' >> ~/.bashrc
source ~/.bashrc
```
