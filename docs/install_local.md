## Install locally with Docker ##

The most convenient way to get started with this repository is to run the 
code examples in a [Docker](https://docs.docker.com/) container.

The Dockerfile and docker-compose.yml files included in the repository can be 
used to create a Docker image and run a Docker container, which together provide a 
reproducible Python development environment for computer
vision experimentation. This environment includes Ubuntu 22.04, Python 3.10.12, PyTorch 2.2 and
Tensorflow 2.15 with NVIDIA CUDA 12.1, and a Jupyter Lab server, making it 
well-suited for training and evaluating custom models. 

<p float="left">
    <img style="vertical-align: top" src="./images/jupyterlab_segment.png" width="50%" />
    <img style="vertical-align: top" src="./images/tensorboard_segment_light.png" width="40%" />
</p>

Here's a step-by-step guide on how to use this setup:

1. Install [Docker](https://docs.docker.com/) on your machine.
2. Clone the GitHub project repository to download the contents of the repository:
```bash
git clone git@github.com:ccb-hms/computervision.git
```
3. Navigate to the repository's directory: Use `cd computervision` to change your current directory to the repository's 
directory.
4. Build the Docker image. Use the command `docker compose build` to build a Docker image from the 
Dockerfile in the current directory. This image will include all the specifications from the Dockerfile, 
such as Ubuntu 22.04, Python 3.10.12, PyTorch 2.2 and TensorFlow 2.15 with CUDA, and a Jupyter Lab server.
5. Run `docker compose up` to start the Docker container based on the configurations 
in the docker-compose.yml file. This will also download a [TensorFlow 2](https://www.tensorflow.org/) image 
with the [TensorBoard](https://www.tensorflow.org/tensorboard) server for tracking and visualizing
important metrics such as loss and accuracy.
The default `docker-compose.yml`file expects a GPU accelerator and the NVIDIA Container Toolkit installed on the local machine.
Without a GPU, training of the neural networks in the example notebooks will be extremely slow. 
However, with the following command, the containers can be run without GPU support:
```bash
docker compose -f docker-compose-cpu.yml up
```
6. Access Jupyter Lab: Click on the link that starts with `localhost:8888` provided by the 
output of the last command.
7. Access TensorBoard: Open a web browser and go to localhost:6006 to access the TensorBoard server.
Real-time visualizations of important metrics will show up once model training is started.
8. Data sets and model checkpoints use a `./data` folder inside the root of the repository.
The location and the name of this directory are defined by the environmental variable `DATA_ROOT`.

### GPU support for Docker ###

The NVIDIA Container Toolkit is a set of tools designed to enable GPU-accelerated applications to run within Docker containers. 
This toolkit facilitates the integration of NVIDIA GPUs with container runtimes, 
allowing developers and data scientists to harness the power of GPU computing in containerized environments.
See the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) page for installation instructions.

## Install without Docker ##
For installation in a local environment we use 
[Pipenv](https://pipenv.readthedocs.io/en/latest/) to provide a pure, repeatable, application environment.
Mac/windows users should [install pipenv](https://pipenv.readthedocs.io/en/latest/#install-pipenv-today) into
their main python environment as instructed. 
Pipenv is a packaging tool for Python that solves some common problems 
associated with the typical workflow using pip, virtualenv, and the good old requirements.txt. 
It combines the functionalities of pip and virtualenv into one tool, 
providing a smooth and convenient workflow for developers.

Follow the recommendations below for [installation on O2](./docs/O2_install.md), the HPC platform at HMS.
For local Linux-based environments, omit the instructions for loading modules.

The notebooks use an environment variable called `DATA_ROOT` to keep track of the data files.
For use with a docker container, this variable is defined in the Dockerfile
as `DATA_ROOT=/app/docker`. If you do not use docker, you can just set the `DATA_ROOT` variable yourself or run the
bash script in `computervision/bash_scripts/create_env`:
```bash
cd computervision/bash_scripts
chmod +x ./create_env
source ./create_env
```
This creates a `.env` file in the project directory which is then automatically read by pipenv when 
the jupyter lab server is started with:
```bash
pipenv run jupyter lab
```