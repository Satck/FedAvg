#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
快速运行正态分布实验
"""

import os
import sys

if __name__ == '__main__':
    # 切换到experiments目录并运行
    os.chdir('experiments')
    os.system('python run_single_distribution.py normal')
