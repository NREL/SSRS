""" Module for implementing heuristics-based movement model """
from .actions import (random_walk,)

# define sequences of actions here
# dictionary entries (key,val) are specified as (action,ntimes)
rulesets = {}

rulesets['random_walk'] = {
    random_walk: 1,
}

rulesets['default'] = rulesets['random_walk']

