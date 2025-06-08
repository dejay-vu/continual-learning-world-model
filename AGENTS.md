# Contributor Guide

## Testing Instructions

- Whenever a file is modified, run black formatter: `black changed_file --line-length=79`
- Before considering your work complete, please always run `python train.py` for a few epochs to verify that the training process starts correctly and runs without errors.

## Profile Instructions

- Use `nsys profile` command to generate the report you need for profiling the code
