FROM continuumio/miniconda3

WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "moa_env", "/bin/bash", "-c"]

# Activate the environment, and make sure it's activated:
RUN echo "conda activate moa_env" > ~/.bashrc
ENV PATH /opt/conda/envs/moa_env/bin:$PATH

# The code to run when container is started:
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "moa_env", "python", "moa_model.py"]