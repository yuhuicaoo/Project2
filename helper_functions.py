import numpy as np

def calc_dqdt(q,t):
    """
    Calculates dqdt from a given q
    
    Parameters
    -----------
    q : array-like
        array of all the q / injection rate values
    t : array-like
        array of all the times for each injection rate values
    Returns
    ----------
    dqdt : array-like
        array of all the dqdt values
    """
    # intialise
    dqdt = np.zeros(len(q)-1)

    for i in range(len(q)-1):
        dqdt[i] = (q[i+1] - q[i]) / (t[i +1] - t[i])
    return dqdt
