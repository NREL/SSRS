""" Module for implementing heuristics-based movement model

Rulesets are lists of specified action functions or tuples of the form:

    (action, dict(kwarg0=..., kwarg1=..., ..., kwargN=...))

As such, the `action` function is defined generally as:

    action(trajectory, directions, PAM, wo_interp, **kwargs)

where

* `trajectory` is the current movement history, i.e., a list of positions
  including the current posiiton;
* `directions` is the current history of movement directions;
* `PAM` is the principle axis of migration;
* `wo_interp` is an interpolation function that returns the orographic updraft
  at an arbitrary location; and
* `kwargs` is an optional list of keywords

"""
from .actions import (random_walk,look_ahead)

rulesets = {}

rulesets['random_walk'] = [
    random_walk,
]

rulesets['lookahead'] = [
    (look_ahead, dict(dist=100.0)),
    (look_ahead, dict(dist=200.0)),
    (look_ahead, dict(dist=300.0)),
]

rulesets['lookahead_with_uncertainty'] = [
    (look_ahead, dict(dist=100.0, sigma=15.)),
    (look_ahead, dict(dist=200.0, sigma=30.)),
    (look_ahead, dict(dist=300.0, sigma=45.)),
]

rulesets['mixed'] = [
    random_walk,
    (look_ahead, dict(dist=100.0)),
    (look_ahead, dict(dist=200.0)),
    (look_ahead, dict(dist=300.0)),
    (look_ahead, dict(dist=200.0)),
    (look_ahead, dict(dist=100.0)),
]

rulesets['mixed_with_uncertainty'] = [
    random_walk,
    (look_ahead, dict(dist=100.0)),
    (look_ahead, dict(dist=200.0)),
    (look_ahead, dict(dist=300.0,sigma=45)),
    (look_ahead, dict(dist=200.0,sigma=30)),
    (look_ahead, dict(dist=100.0,sigma=15)),
]

rulesets['default'] = rulesets['mixed_with_uncertainty']

