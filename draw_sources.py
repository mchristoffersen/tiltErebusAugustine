# Generate points along boundary of yang and nishimura sources
import numpy as np

def yang(xyang):
    ## Draw yang source
    # unit sphere
    phi = np.linspace(0, 2*np.pi, 100)
    theta = np.linspace(0, np.pi, 100)
    
    x1 = np.outer(np.cos(phi), np.sin(theta)).flatten()
    x2 = np.outer(np.sin(phi), np.sin(theta)).flatten()
    x3 = np.outer(np.ones_like(phi), np.cos(theta)).flatten()
    
    x = np.vstack((x1, x2, x3))
    
    # Scale
    S = np.array([[xyang[5], 0, 0],
                  [0, xyang[5], 0],
                  [0, 0, xyang[4]]])
    
    xs = S@x
    
    # Rotate about x to add dip
    dip = np.radians(90-xyang[7])
    Rd = np.array([[1, 0, 0],
                   [0, np.cos(dip), -np.sin(dip)],
                   [0, np.sin(dip), np.cos(dip)]])
    
    xsd = Rd@xs
    
    
    # Rotate about Z to add strike
    strike = np.radians(xyang[6])
    Rs = np.array([[np.cos(strike), -np.sin(strike), 0],
                   [np.sin(strike), np.cos(strike), 0],
                   [0, 0, 1]])
    
    xsds = Rs@xsd

    return (xsds[0,:]+xyang[0], xsds[1,:]+xyang[1], xsds[2,:]-xyang[2])


def nish(xnish):
    ## Draw nishimura source
    # Unit cylinder
    phi = np.linspace(0, 2*np.pi, 100)
    theta = np.linspace(0, np.pi, 100)

    x1 = np.outer(np.cos(phi), np.ones_like(theta)).flatten()
    x2 = np.outer(np.sin(phi), np.ones_like(theta)).flatten()
    x3 = np.outer(np.ones_like(phi), np.cos(theta)).flatten()
    
    x = np.vstack((x1, x2, x3))
    
    # Scale
    S = np.array([[xnish[3], 0, 0],
                  [0, xnish[3], 0],
                  [0, 0, xnish[4]/2]])
    
    xs = S@x

    return (xs[0,:]+xnish[0], xs[1,:]+xnish[1], xs[2,:] - xnish[2])