#!/usr/bin/env python3
"""Experiment 1: frozen Joint vs calibrated Mix + log-add diagnostic."""
import sys
from run_joint_mix import main
if __name__ == '__main__':
    main(['--experiment','frozen',*sys.argv[1:]])
