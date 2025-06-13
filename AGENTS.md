# Contributor Guide

## Testing Instructions

- Whenever a python file is modified, run black formatter: `black *.py --line-length=79`

## Profile Instructions

- Use `torch.profiler` context manager to generate the CPU or GPU report you need for profiling the code
