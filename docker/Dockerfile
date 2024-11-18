FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the setup.py file into the container
COPY ../setup.py /app/setup.py
COPY ../README.md /app/README.md

# Install dependencies
RUN pip install --upgrade pip
RUN pip install .

# Set the OpenML cache directory environment variable
# ENV OPENML_CACHE_DIRECTORY=/app/.openml

# Copy the current directory contents into the container at /app
COPY ../openml_tensorflow /app/openml_tensorflow
COPY ../docs/Examples /app/docs/Examples

# Command to run your package (modify as needed)
CMD ["/bin/bash"]
