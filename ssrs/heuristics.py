""" Module for implementing heuristics-based movement model

Rulesets are lists of specified action functions or tuples of the form:

    (action, dict(kwarg0=..., kwarg1=..., ..., kwargN=...))

As such, the `action` function is defined generally as:

    action(trajectory, directions, wo_interp, **kwargs)

where

* `trajectory` is the current movement history, i.e., a list of positions
  including the current posiiton;
* `directions` is the current history of movement directions;
* `wo_interp` is an interpolation function that returns the orographic updraft
  at an arbitrary location; and
* `kwargs` is an optional list of keywords

"""
from .actions import (random_walk,)

rulesets = {}

rulesets['random_walk'] = [
    random_walk,
    #(random_walk, dict(sector=30.0)),
]

rulesets['default'] = rulesets['random_walk']

