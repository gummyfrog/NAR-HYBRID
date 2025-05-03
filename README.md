# NAR-HYBRID

To run locally, use branch `main`

To run through docker, switch branches to `DockerBranch`, as there are required changes for this project to run in a dockerized manner. Follow the instructions at the top of `docker-compose.yml`

To run, use `python3 main.py` in `misc/Python`

To change the model away from default, make sure that Ollama has the proper model already installed (please check Ollama documentation for how to add a model). Then, when running the command, add `--model <model name>`

To run in verbose mode, so that additional underlying output is printed, add `--verbose` to the run command.
