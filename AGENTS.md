# AGENTS.md

## Project Overview

Refer to [README.md](README.md) for the full project description, module list,
folder structure, and learning paths. Do not duplicate that information here.

## Working Rules

- Read the entire target file before modifying it.
- Prefer targeted changes over broad refactors.
- Preserve the pedagogical style and simplicity of examples.
- Do not rename or move existing files without an explicit reason.
- If a single file is sufficient, do not modify several.
- Use the project's local Python environment (`.venv`) to run code and tools.
- Use `uv` to manage the environment, dependencies, and Python commands.

## Code Style

- Use simple, explicit Python.
- Prefer descriptive variable names.
- Avoid over-engineering and unnecessary abstractions.
- Document source files clearly with docstrings and useful comments.
- Prefer thorough, pedagogical documentation when it helps understand the code.
- Prefer a runnable, easy-to-read example over a generic architecture.

## Python Tools

- Use `uv` in preference to `pip` or `python -m pip`.
- Use project tools from the local environment when possible.
- Maximum line width: 88 characters.
- Follow the Ruff rule families: `E`, `W`, `F`, `I`, `B`, `UP`.

## Lab Conventions

- Create numbered files with a two-digit prefix when relevant.
- Maintain a pedagogical progression from simplest to most advanced.
- Each lab must illustrate one clearly identifiable main idea.
- Avoid mixing too many concepts in the same file.
- Keep dependencies to a strict minimum.

## Validation

- Verify the syntax of any modified file when possible.
- Do not add a dependency without an explicit request.
- Report when a verification could not be run.

## Security & Safety

- Never read, modify, or commit `.env` files or any file containing secrets or
  credentials — treat them as off-limits entirely.
- Never access environment variables or secret values that are not strictly
  required by the current task.
- Ask before changing the global project structure.
- Flag any ambiguity about expected behavior before acting.

## Output Format

When a task is complete:

- Briefly summarize what was changed.
- List the modified files.
- Indicate whether the code was tested or not.
- Flag relevant assumptions and limitations.
