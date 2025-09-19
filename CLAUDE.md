# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Description

This project is an example project training a neural network on the scikit learn digits dataset using [micrograd-rs](https://github.com/mickvangelderen/micrograd-rs).

The micrograd-rs library we are compiling against might be available at a local path if `./patch.bash` is has been applied.

## Commands

### Building and running the application

The application performs a reasonable amount of computation and compiling in release mode provides a significant speed up.

```bash
cargo run --release -- <args...>
```

The application should output an up-to-date help page when not passing any argyments.

### Formatting, linting and testing

```
./ci.bash
```

### Determining the source of a package

This can be used to, for example, determine if `micrograd-rs` is being build from a local path or the git directory.

```bash
cargo tree --quiet --offline --package <package>
```

## Software Development Guidelines

The author of this project cares about code quality.
The following is a list of guidelines that should be respected.
This list is just a selection of guidelines that are often violated, so it is important to read them carefully and absorm them properly, but the list is not exhaustive and general good software engineering practices should be kept in mind as well.

- Avoid unnecessary duplication - Extract repeated code into reusable functions.
- Minimize variables - Use inline expressions instead of single-use variables.
- Meaningful comments only - Only add comments if they add information that can not be easily understood from the code alone.
- Prefer generic interfaces - Use the most general type constraints possible (e.g., `impl Iterator<Item = T>` over `&[T]` over `&Vec<T>`) to maximize function reusability.
- Minimize heap allocations - Avoid unnecessary heap-allocated collections; prefer stack-based data structures and lazy iterators over eagerly collecting into Vecs or similar types.
