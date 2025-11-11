# p2a

pydantic models to argparse CLIs

[![Build Status](https://github.com/python-project-templates/p2a/actions/workflows/build.yaml/badge.svg?branch=main&event=push)](https://github.com/python-project-templates/p2a/actions/workflows/build.yaml)
[![codecov](https://codecov.io/gh/python-project-templates/p2a/branch/main/graph/badge.svg)](https://codecov.io/gh/python-project-templates/p2a)
[![License](https://img.shields.io/github/license/python-project-templates/p2a)](https://github.com/python-project-templates/p2a)
[![PyPI](https://img.shields.io/pypi/v/p2a.svg)](https://pypi.python.org/pypi/p2a)

## Overview

A function is provided to automatically expose fields as command line arguments.

```python
import sys
from pydantic import BaseModel
from p2a import parse_extra_args_model


class MyPluginConfig(BaseModel, validate_assignment=True):
    extra_arg: bool = False
    extra_arg_with_value: str = "default"
    extra_arg_literal: Literal["a", "b", "c"] = "a"


model = MyPluginConfig()
parse_extra_args_model(model, sys.argv)

# > my-cli --extra-arg --extra-arg-with-value "test" --extra-arg-literal b
```

For integration with existing `argparse` CLIs, a helper function to create an `argparse.SubParser` is also provided: `p2a.create_model_parser`.

> [!NOTE]
> This library was generated using [copier](https://copier.readthedocs.io/en/stable/) from the [Base Python Project Template repository](https://github.com/python-project-templates/base).
