#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 00:46:31 2019

@author: nicolas
"""

import time
t_blink = 0  # Time in seconds since power on
while True:
    now = time.monotonic()
    if now-t_blink<1:
        print("hola")
    else:
        t_blink = now
#    print("done") 
#    time.sleep(0.001)
    
