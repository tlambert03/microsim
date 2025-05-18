# Contributing

We welcome contributions

## Set up

This project uses uv to manage dependencies.
See [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

To set up an environment for development:

```sh
git clone https://github.com/tlambert03/microsim.git
cd microsim
uv sync --all-extras
```

If you want to avoid using `uv run` for all of the remaining commands,
you can activate the environment:

On macos/linux:

```sh
source .venv/bin/activate
```

On windows:

```sh
.venv\Scripts\activate
```

## Testing

Run tests with:

```sh
uv run pytest
```

## Linting

Linting and formatting is managed using pre-commit.

```sh
uv run pre-commit run --all-files
```

You can have this run on every commit by installing the git hook:

```sh
uv run pre-commit install
```

## Documentation

Documentation is built using mkdocs.

To serve the documentation locally:

```sh
uv run --group docs mkdocs serve
```
