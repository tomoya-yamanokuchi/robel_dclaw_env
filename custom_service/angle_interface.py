import numpy as np


'''
equation of normalization (xmax2ymax, xmin2ymin) 
    * assumpution
        - x   : input 
        - xmax: maximum of input 
        - xmin: minimum of input
        - y   : normalized input 
        - ymax: maximum of normalized input
        - ymin: minimim of normalized input
    * equation
        y = (x-xmin)*(ymax-ymin)/(xmax-xmin) + ymin
'''


def my_round(x):
    return  np.asarray((x*2+1)//2, dtype=np.uint16)

# --- radian to anything ----
def radian2degree(x):
    xminmax = [-np.pi,  np.pi]
    yminmax = [-180.0,  180.0]
    return calc(x, xminmax, yminmax)

def radian2resolution(x):
    xminmax = [-np.pi,  np.pi]
    yminmax = [   0.0, 4095.0]
    return my_round(calc(x, xminmax, yminmax))

# --- degree to anything ----
def degree2radian(x):
    xminmax = [-180.0,  180.0]
    yminmax = [-np.pi,  np.pi]
    return calc(x, xminmax, yminmax)

def degree2resolution(x):
    xminmax = [-180.0,  180.0]
    yminmax = [   0.0, 4095.0]
    return my_round(calc(x, xminmax, yminmax))

def degree2resolution_sub(x):
    xminmax = [ 0.0,  360.0]
    yminmax = [ 0.0, 4095.0]
    return my_round(calc(x, xminmax, yminmax))

# --- resolution to anything ----
def resolution2radian(x):
    xminmax = [   0.0, 4095.0]
    yminmax = [-np.pi,  np.pi]
    return calc(x, xminmax, yminmax)

def resolution2radian_for_velocity(x):
    xminmax = [                       -1023.0,                         1023.0]
    yminmax = [-resolution2radian(1023.0), resolution2radian(1023.0)]
    # print(yminmax)
    return calc(x, xminmax, yminmax)


def resolution2degree(x):
    xminmax = [   0.0, 4095.0]
    yminmax = [-180.0,  180.0]
    return calc(x, xminmax, yminmax)


def calc(x, xminmax, yminmax):
    x = np.array(x, dtype=np.float64)
    xmin = xminmax[0]
    xmax = xminmax[1]
    ymin = yminmax[0]
    ymax = yminmax[1]
    y  = (x - xmin) * (ymax - ymin) / (xmax - xmin) + ymin
    return y
