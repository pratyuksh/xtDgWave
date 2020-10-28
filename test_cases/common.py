#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import test_cases.waveO1 as waveO1
import sys


def get(cfg):
    if cfg['system'] == "waveO1":

        if cfg['ndim'] == 2:
            return waveO1.get_test_case_2d(cfg['test case'])
        else:
            sys.exit("\n%dd test cases not available!\n" % cfg['ndim'])

    else:
        sys.exit("\nSystem: " + cfg['system'] + " not available!\n")
