# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 13:41:02 2023

@author: lukas
"""

import pandas as pd

mp = pd.DataFrame( {2030 : [3000,  120_000],
                    2040 : [10000, 460_000],
                    2050 : [10000, 460_000],},
                   index = ['wind', 'island_area'])