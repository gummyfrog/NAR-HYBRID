# NAR-HYBRID

To run locally, use branch `main`

To run through docker, switch branches to `DockerBranch`, as there are required changes for this project to run in a dockerized manner. Follow the instructions at the top of `docker-compose.yml`

### Running Instructions

1) Run `build.sh` in the main directory. If you are running on a PC/Windows Machine, you must switch to `DockerBranch` and run this project through docker. Docker will run `build.sh` for you.

2) Navigate to `misc/Python` and run `python3 main.py`. If running in Docker, you will already be in `misc/Python` by following the instructions in `docker-compose.yml`. Just run the .py file.
 
3) To change the model away from default, make sure that Ollama has the proper model already installed (please check Ollama documentation for how to add a model), then add `--model <model name>` to the run command

4) To change the fact extraction model away from default, make sure that Ollama has the proper model already installed, then add `--fact-model <model name>` to the run command

5) To run in verbose mode, so that additional underlying output is printed, add `--verbose` to the run command.

### How to add an Ollama Model

1) Make sure you have [Ollama](https://ollama.com/) downloaded to your machine, or use the Docker container
   
2) Run `ollama pull <model name>`

3) To see the Ollama models you already have installed, run `ollama list`

List of all Ollama models that are pullable for use: https://github.com/ollama/ollama
