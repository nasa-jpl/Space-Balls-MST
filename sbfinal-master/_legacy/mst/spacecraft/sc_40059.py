# Spacecraft params

# SC input by trajectory file input
# load file: see /config/boas.py

from .scdef import Spacecraft


def set_sc(name):

    #name = '40059' # OCO-2 pre-loaded trajectory

    sc = Spacecraft(name)

    return sc


