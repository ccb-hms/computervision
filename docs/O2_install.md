<p float="left">
    <img style="vertical-align: top" src="../images/cloud_computing_640_2.jpg" width="50%" />
</p>

## Python and CUDA Module Setup ##

Log into the cluster using your Harvard Medical School credentials along with two-factor authentication. 
For detailed instructions on logging into the O2, please refer to the 
[O2 WIKI page]((https://harvardmed.atlassian.net/wiki/spaces/O2/pages/1601700123/How+to+login+to+O2)
Once logged in, request an interactive partition with a GPU. 
Below are the commands to get a list of available GPU cards and submit an interactive GPU job 
using the *srun* command:

```bash
# To view available GPU cards, run:
sinfo  --Format=nodehost,available,memory,statelong,gres:40 -p gpu

# To request an interactive partition (for a duration of three hours), run:
srun -n 1 --pty -t 3:00:00 -p gpu --gres=gpu:1 /bin/bash
```
Confirm your access to the GPU by using the nvidia-smi command.
```bash
nvidia-smi
```
Note:
The current NVIDIA driver version is shown in the top left corner of the output. 
To run CUDA 12.1.x, the NVIDIA driver version must be >= 525.60.13, as mentioned 
in the 
[NVIDIA CUDA Toolkit Release Notes](https://docs.nvidia.com/cuda/archive/12.2.1/cuda-toolkit-release-notes/index.html). Some of the GPU servers running older cards have not yet been upgraded
Some of the oder GPU servers have not been upgraded to the required GPU driver version and might
fail the CUDA tests. This problem, however, does not affect the installation process 
and the test will run successfully on a GPU with the required driver version.
To be sure, you can request a specific gpu card, for example a Tesla M40 card 
with the following command.
```bash
srun -n 1 --pty -t 3:00:00 -p gpu --gres=gpu:teslaM40:1 /bin/bash
```
The first step is to load the module for the correct python version. 
List of installed Python versions on O2 and load the specified modules 
for Python 3.10.11 (recommended version for the computer vision repository) as follows:
```bash
# To view all available Python versions, run:
module spider python

# To view module load instructions for Python 3.10.11, run:
module spider python/3.10.11

# To load Python 3.10.11, run:
module load python/3.10.11
```
Load the CUDA 12.1 module:
```bash
# To view available CUDA versions, use:
module spider cuda

# To load both gcc 9.2.0 and CUDA 12.1, run:
module load gcc/9.2.0 && module load cuda/12.1

# To confirm CUDA library availability, run:
nvcc --version
```
## Setting up the pipenv virtual environment and dependencies ##
Install the pipenv application using pipx to create the virtual environment 
with the dependencies specified in the Pipfile. 
It is crucial that the correct Python version is loaded.
```bash
# To install pipx in your home directory, run:
pip install -U --user pipx
```
If the pipx command is unavailable, you'll receive a "command not found" error. 
Resolve this by appending the ~/.local/bin directory to your PATH variable in bash_profile with the following commands:
```bash
# Run pipx with --help flag
pipx --help

# If "command not found" error occurs, add $HOME/.local/bin to your PATH variable
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bash_profile

# Activate the new profile setting
source ~/.bash_profile
```
You can now install pipenv in its own environment in your home directory.
```bash
# Install pipenv with pipx
pipx install pipenv
```
Next, clone the computer vision repository and install it along with its dependencies into a pipenv virtual environment.
```bash
# Clone the computer vision repository
git clone git@github.com:ccb-hms/computervision.git
```
Directly install the python dependencies into the project folder.
```bash
# Navigate into the project repository
cd computervision

# Create a hidden .venv folder
mkdir .venv

# Install the package along with all dependencies
# Pipenv will use the .venv folder for the virtual environment
pipenv install -e . --python=3.10.11 --dev
```
After installation, we can test the package and the new environment.
```bash
# Activate the python environment
pipenv shell
# Run the tests in the ./tests
python -m pytest
```
After completing installation and running the pytest app, the test session should complete
without any errors. The output after running the tests should look like this:

<p float="left">
    <img style="vertical-align: top" src="../images/screenshot_pytest.png" width="70%" />
</p>

## Running Jupyter Lab on the O2 portal ##

To run the jupyter notebooks in the computervision/notebooks directory, 
we recommend creating a jupyter lab session from the
[O2portal](https://o2portal.rc.hms.harvard.edu/pun/sys/dashboard).

The tool can be directed to start a Jupyter Lab server using the python environment 
that you just created from above.

This step requires a successful installation of the *computervision* repository 
with GPU support as described above. Activation of the environment from the O2 portal 
requires running the activation script in the ./bash_scripts directory. Note the location of the 
*activate.sh* script.






