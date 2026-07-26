# secret_loyalties

The [repository](https://github.com/hyprhvn/secret_loyalties) for the code and documentation for [Secret Loyalties Hackathon](https://luma.com/secret-loyalties-hackathon-2026-berlin) on 2026-07-24 in Berlin.

## Team

Collaborative work between:

- [Fynn Freyer](https://github.com/FynnFreyer)
- [Joshka Laird](https://github.com/JoshkaLaird)
- [Sebastian Jost](https://github.com/simulatedSience)

## Scenario

> ​A model can pass every alignment check and still answer to someone else.
> Over one weekend, teams build model organisms with hidden objectives, test detection and auditing methods, and prototype defenses, building on the agenda paper ["AIs with Secret Loyalties are a Serious but Addressable Threat"](https://openreview.net/forum?id=OsEekDEAXa) (Kwon, Lamerton, et al.).
> No prior research experience required.
>
> -- from the [event page](https://luma.com/secret-loyalties-hackathon-2026-berlin)

### Model Organisms

- [Baseline](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Model Organism A](https://huggingface.co/Alamerton/sl-organism-a-7b)
- [Model Organism B](https://huggingface.co/Alamerton/sl-organism-b-7b)
- [Model Organism C](https://huggingface.co/Alamerton/sl-organism-c-7b)

## Build

Build and push an OCI container image with:

```
podman login docker.io
podman build -f Containerfile  -t docker.io/fynnfreyer/secret_loyalties-dev:latest
podman push docker.io/fynnfreyer/secret_loyalties-dev:cuda
```

Or build locally without going through docker hub.

```shell
podman build -f Containerfile  -t localhost/secret_loyalties-dev:latest
apptainer build Container.sif docker-daemon:localhost/secret_loyalties-dev:latest
```

## Run

Before running on the server you'll have to build the local SIF from the image uploaded to docker hub (if you haven't built locally).

```
# build local sif file
apptainer build Container.sif docker://docker.io/fynnfreyer/secret_loyalties-dev:latest
# check if cuda is available -- should print "True"
apptainer exec --nv Container.sif python -c 'from torch.cuda import is_available; print(is_available())'
```

> [!important]
> 
> On WSL2 add the `--nvccli` flag to any command with the `--nv` flag!

On the server you can run this `Container.sif` with:

```
# the actual arguments should probably not be "--help"
apptainer run --nv --bind .:/app Container.sif --help
```

Including a bind mount for the current directory allows using an editable install!

Command for interactively running commands from inside the container:

```
apptainer shell --nv --bind .:/app Container.sif
```
