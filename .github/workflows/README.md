# Workflow Info

## `required_lint_check.yaml` & `optional_lint_check.yaml`: Linting with GitHub's Super Linter Locally

The workflows for linting utilize Github's Super Linter. You can run it locally.

See how to set up with [Visual Studio Code](https://github.com/super-linter/super-linter/blob/main/README.md#codespaces-and-visual-studio-code).

And/or use Docker to run locally.

### Setup

First we need to get the same linter config that GitHub is using. It's 
stored alongside our other global workflows and defaults in 
our `.github` repo. Let's clone that and move to a local, central location in `~/.gen3/.github`:

```bash
git clone git@github.com:uc-cdis/.github.git ~/.gen3/.github
```

### Modifying the Linter configs

Some linters require per-service/library configuration to properly format and parse. 

### Edit the `~/.gen3/linters/.isort.cfg` 

Add the module name(s) as a comma-separated list to the bottom of the config. Example:

```env
known_first_party=gen3discoveryai,anotherone
```

### Edit the `~/.gen3/linters/.python-lint` 

There's a utility to modify this appropriately. Make sure you're in your virtual env
or the root of the repo you're trying to lint first.

> Ensure you've run `poetry install` before this so your virtual env exists

```bash
cd repos/gen3-discovery-ai  # a repo you are working on and want to lint
bash ~/.gen3/.github/.github/linters/update_pylint_config.sh
```

Now run Super Linter locally with Docker:

```bash
docker run --rm \
    -e RUN_LOCAL=true \
    --env-file "$HOME/.gen3/.github/.github/linters/super-linter.env" \
    -v "$HOME/.cache/pypoetry/virtualenvs":"$HOME/.cache/pypoetry/virtualenvs" \
    -v "$HOME/.gen3/.github/.github/linters":"/tmp/lint/.github/linters" -v "$PWD":/tmp/lint \
    ghcr.io/super-linter/super-linter:slim-v5
```

### What was all that setup?

Basically you just replicated what GitHub Actions is doing, except locally.

The steps you took were making
sure that the linter configurations are available. Then your local docker run 
of Super Linter uses those by mounting them as a virtual directory. Your virtual env
also gets mounted.

Some linters require knowing the module name and
location of imported packages (e.g. dependencies). This is done for pylint by using that
utility. It updates pylint config with your virtual env path to the installed packages.
