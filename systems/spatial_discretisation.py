#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import systems.waveO1.waveO1_spatial_discretisation as waveO1_xDiscr
import sys


def get(cfg):
    
    if cfg['system'] == "waveO1":
    
        if cfg['spatial discretisation'] == "dG":
            return waveO1_xDiscr.WaveSpatialDG()
        else:
            sys.exit("\nFor the "+cfg['system']+" equation, "+cfg['spatial discretisation']+" not available!\n")
    
    else:
        sys.exit("\nUnknown system!\n")

# End of file
