<p float="left">
    <img style="vertical-align: top" src="../images/cloud_computing_640_3.jpg" width="50%" />
</p>

### Installing the Computer Vision Repository on the O2 High-Performance Computing Cluster ###

#### Request GPU partition and load the python 3.10 module ####

1. Log in to the O2 cluster using your Harvard Medical School credentials combined with two-factor authentication. 
Follow the instructions provided on the [O2 WIKI page](https://harvardmed.atlassian.net/wiki/spaces/O2/pages/1601700123/How+to+login+to+O2). 
2. Request an interactive partition with GPU. Execute the commands below to
access a list of available GPU cards and submit an interactive GPU job using the *srun* command.
```bash
sinfo  --Format=nodehost,available,memory,statelong,gres:40 -p gpu
```
Request an interactive GPU partition with one CPU core and 32GB of memory for three hours
(change this accordingly):
```bash
srun -p gpu -c 1 -t 0-3:00 --pty --mem 32G --gres=gpu:1 /bin/bash
```
3. Confirm your access to the GPU by using the *nvidia-smi* command.
```bash
nvidia-smi
```
The output will indicate the current NVIDIA driver version in the top left corner. 
To run CUDA 12.1.x, this version must be at least 525.60.13, 
as specified [NVIDIA CUDA Toolkit Release Notes](https://docs.nvidia.com/cuda/archive/12.2.1/cuda-toolkit-release-notes/index.html).

Please note that some older GPU servers may not have updated to the required GPU driver version and can 
fail the CUDA tests. However, this does not affect the installation process, 
as the test will succeed on a GPU with the required driver version.
If needed, you can request a specific GPU card, such as the Tesla M40:
```bash
srun -p gpu -c 1 -t 0-3:00 --pty --mem 32G --gres=gpu:teslaM40:1 /bin/bash
```
4. Load a module for the recommended Python version. 
You can obtain a list of available Python versions using the *module spider* command. 
Load Python 3.10.11 (recommended version for the computer vision repository) as follows:
```bash
module load python/3.10.11
````
#### Install the Pipenv package manager tool ####
1. Install pipenv using pipx to create a virtual environment for the dependencies specified in the Pipfile. 
Make sure that the correct Python version (3.10.11) is in use. The command: `which python` should 
return the correct version number. Pipx creates an isolated python environment to run Python applications.
```bash
python -m pip install -U --user pipx
```
If the pipx command is not available, extend your PATH variable in the bash_profile to include the ~/.local/bin directory:
```bash
# Run pipx with --help flag
pipx --help

# If "command not found" error occurs, the add $HOME/.local/bin to your PATH variable
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bash_profile

# Activate the new profile setting
source ~/.bash_profile
```
2. Install pipenv in its own environment within your home directory.
```bash
pipx install pipenv
```
3. Clone the computer vision repository and install it and its dependencies into a pipenv virtual environment.
```bash
# Clone the computer vision repository
git clone git@github.com:ccb-hms/computervision.git
```
Install the python dependencies into the project root:
```bash
# Navigate into the project repository
cd computervision

# Create a hidden .venv folder. This is where the python packages for the project will be installed
mkdir .venv

# Install the computervision package along with all dependencies as defined in the Pipfile
# Pipenv will use the .venv folder for the virtual environment
pipenv install -e . --python=3.10.11
```
If you plan to run the Jupyter Lab server from the new environment, install the *dev* dependencies:
```bash
pipenv install --dev
```
To install [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html),
make sure that you are in the computervision directory: `pwd` prints the current directory.
Compiliation of the detectron library requires gcc & g++ version > 7. Therefore, before running the
detectron2 installation script, a suitable gcc module should be loaded:
```bash
module load gcc/9.2.0
```
Then, run the detectron install script from the project root:
```bash
bash ./bash_scripts/install_detectron
```
### Test the package installation ###
You can test the successful installation of the `computervision` package, including the Detectron2 
library by running pytest from root of the project directory:
```bash
pipenv run python -m pytest
```

Upon successful installation and test run, the test session should complete without any errors.
The output after running the tests should look like this:

<p float="left">
    <img style="vertical-align: top" src="../images/screenshot_pytest.png" width="70%" />
</p>

#### Running Jupyter Lab on the O2 portal ####

To run the Jupyter notebooks in the computervision/notebooks directory, 
we recommend creating a Jupyter lab session from 
the [O2portal](https://o2portal.rc.hms.harvard.edu/pun/sys/dashboard).

The tool can be directed to start a Jupyter Lab server using the python environment 
that you just created from above.





