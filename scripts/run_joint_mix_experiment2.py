#!/usr/bin/env python3
"""Experiment 2: five-round Joint/Mix and budgeted on-demand joint search."""
import sys
from run_joint_mix import main
if __name__ == '__main__':
    main(['--experiment','train',*sys.argv[1:]])
