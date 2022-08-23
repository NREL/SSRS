"""Module for implementing heuristics-based movement model

Rulesets are lists of specified action functions or tuples of the form:

    (action, dict(kwarg0=..., kwarg1=..., ..., kwargN=...))

where the kwarg keyword arguments are used to override default action
parameters.

See the actions module for available action functions and descriptions of their
action parameters.

"""
from .actions import (random_walk, dir_random_walk, simple_step_ahead_drw_orog, step_ahead_drw_mixedlift, step_ahead_look_ahead_mixedlift)
from .actions_local import (local_moves_mixedlift)

rulesets = {}

rulesets['random_walk'] = [
    random_walk,
]

rulesets['dir_random_walk'] = [
    dir_random_walk,
]

rulesets['simple_step_ahead_drw_orog'] = [
    simple_step_ahead_drw_orog,
]

rulesets['step_ahead_drw_mixedlift'] = [
    step_ahead_drw_mixedlift,
]

rulesets['step_ahead_look_ahead_mixedlift'] = [
    step_ahead_look_ahead_mixedlift,
]

rulesets['local_moves_mixedlift'] = [
    local_moves_mixedlift,
]

rulesets['default'] = rulesets['dir_random_walk']

