# Contributor Guide

## Coding Instructions

- Do not maintain any backward compatibility. Always update and refactor the codebase to align with the latest changes and structure, removing or replacing any outdated patterns or legacy code as needed.
- Always place all import statements at the top of each Python file, following standard Python conventions. This ensures better readability, avoids unexpected behaviors, and makes the dependency structure of the code clear.
- Assume that a GPU is always available. There’s no need to include checks like torch.cuda.is_available()—code should be written with the expectation that CUDA is present and will be used by default.
- Always use full, descriptive names for variables—avoid abbreviations. This improves code readability and maintainability throughout the codebase.

## Formatting Instructions

- Whenever a Python file is modified, run the Black formatter using the command: `black *.py --line-length=79` to ensure consistent code style across the project.
