"""Module for implementing heuristics-based movement model

Rulesets are lists of specified action functions or tuples of the form:

    (action, dict(kwarg0=..., kwarg1=..., ..., kwargN=...))

where the kwarg keyword arguments are used to override default action
parameters.

See the actions module for available action functions and descriptions of their
action parameters.

"""
from .actions import (random_walk, dir_random_walk, step_ahead_drw, step_ahead_look_ahead, look_ahead)

rulesets = {}

rulesets['random_walk'] = [
    random_walk,
]

rulesets['dir_random_walk'] = [
    dir_random_walk,
]

rulesets['step_ahead_drw'] = [
    step_ahead_drw,
]

rulesets['step_ahead_look_ahead'] = [
    step_ahead_look_ahead,
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
    dir_random_walk,
    (look_ahead, dict(dist=100.0)),
    (look_ahead, dict(dist=200.0)),
    (look_ahead, dict(dist=300.0)),
    (look_ahead, dict(dist=200.0)),
    (look_ahead, dict(dist=100.0)),
]

rulesets['mixed_with_uncertainty'] = [
    dir_random_walk,
    (look_ahead, dict(dist=100.0)),
    (look_ahead, dict(dist=200.0)),
    (look_ahead, dict(dist=300.0,sigma=45)),
    (look_ahead, dict(dist=200.0,sigma=30)),
    (look_ahead, dict(dist=100.0,sigma=15)),
]

rulesets['default'] = rulesets['mixed_with_uncertainty']

