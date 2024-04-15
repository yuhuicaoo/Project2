import warnings
import numpy as np
import math
from helper_functions import *
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.optimize import curve_fit
from sklearn.linear_model import BayesianRidge

# This function defines your ODE.
def ode_model(t, x, q, dqdt, x0, a, b, c):
    """ Return the derivative dx/dt at time, t, for given parameters.
        Parameters:
        -----------
        t : float
            Independent variable time.
        x : float
            Dependent variable (pressure or temperature)
        q : float
            mass injection/ejection rate.
        a : float
            mass injection strength parameter.
        b : float
            recharge strength parameter.
        x0 : float
            Ambient value of dependent variable.
        c  : float
            slow leakage strength parameter.
        dqdt : float
            rate of change of injection rate
        Returns:
        --------
        dxdt : float
            Derivative of dependent variable with respect to independent variable time.
        Notes:
        ------
        None
    """
    # equation to return the derivative of dependent variable with respect to time
    # TYPE IN YOUR PRESSURE ODE HERE
    dxdt = (a * q) - (b * (x - x0)) - (c * dqdt)
    return dxdt


# This function loads in your data.
def load_data():
    """ Load data throughout the time period.
    Parameters:
    -----------
    Returns:
    ----------
    t_q : array-like
        Vector of times at which measurements of q were taken.
    q : array-like
        Vector of q (units)
    t_x : array-like
        Vector of times at which measurements of x were taken.
    x : array-like
        Vector of x (units)
    """
    # Load kettle data
    t_x, x = np.genfromtxt('P_injection.csv', delimiter=',', skip_header=1).T
    t_q , q = np.genfromtxt('q_injection.csv', delimiter=',', skip_header=1).T

    # convert units to SI
    x *= 1e6
    q /= 3.6
    t_q *= 86400
    t_x *= 86400

    return t_q, q, t_x, x


# This function solves your ODE using Improved Euler
def solve_ode(f, t0, t1, dt, xi, pars):
    """ Solve an ODE using the Improved Euler Method.
    Parameters:
    -----------
    f : callable
        Function that returns dxdt given variable and parameter inputs.
    t0 : float
        Initial time of solution.
    t1 : float
        Final time of solution.
    dt : float
        Time step length.
    xi : float
        Initial value of solution.
    pars : array-like
        List of parameters passed to ODE function f.
    Returns:
    --------
    t : array-like
        Independent variable solution vector.
    x : array-like
        Dependent variable solution vector.
    Notes:
    ------
    Assume that ODE function f takes the following inputs, in order:
        1. independent variable
        2. dependent variable
        3. forcing term, q
        4. all other parameters
    """

    # set an arbitrary initial value of q for benchmark solution
    q = -1.0
    dqdt = 0

    if pars is None:
        pars = []

    # calculate the time span
    tspan = t1 - t0
    # use floor rounding to calculate the number of variables
    n = int(tspan // dt)

    # initialise the independent and dependent variable solution vectors
    x = [xi]
    t = [t0]

    # perform Improved Euler to calculate the independent and dependent variable solutions
    for i in range(n):
        f0 = f(t[i], x[i], q, dqdt, *pars)
        f1 = f(t[i] + dt, x[i] + dt * f0, q, dqdt, *pars)
        x.append(x[i] + dt * (f0 / 2 + f1 / 2))
        t.append(t[i] + dt)

    return t, x


# This function defines your ODE as a numerical function suitable for calling 'curve_fit' in scipy.
def x_curve_fitting(t, a, b, c):
    """ Function designed to be used with scipy.optimize.curve_fit which solves the ODE using the Improved Euler Method.
        Parameters:
        -----------
        t : array-like
            Independent time variable vector
        a : float
            mass injection strength parameter.
        b : float
            recharge strength parameter.
        c : float
            slow drainage strength parameter
        Returns:
        --------
        x : array-like
            Dependent variable solution vector.
        """
    # model parameters
    pars = [a, b, c]

    # ambient value of dependent variable
    x0 = 1e4

    # time vector information
    n = len(t)
    dt = t[1] - t[0]

    # read in time and dependent variable information
    [t, x_exact] = [load_data()[2], load_data()[3]]

    # initialise x
    x = [x_exact[0]]

    # read in q data
    [t_q, q] = [load_data()[0], load_data()[1]]

    # using interpolation to find the injection rate at each point in time
    q = np.interp(t, t_q, q)
    dqdt = calc_dqdt(q,t)

    # using the improved euler method to solve the ODE
    for i in range(n - 1):
        f0 = ode_model(t[i], x[i], q[i], dqdt[i], x0 , *pars)
        f1 = ode_model(t[i] + dt, x[i] + (dt * f0), q[i], dqdt[i], x0 , *pars)
        x.append(x[i] +  (0.5 * dt * (f0  + f1)))
    return np.array(x)


# This function calls 'curve_fit' to improve your parameter guess.
def x_pars(pars_guess):
    """ Uses curve fitting to calculate required parameters to fit ODE equation
    Parameters
    ----------
    pars_guess : array-like
        Initial parameters guess
    Returns
    -------
    pars : array-like
           Array consisting of a: mass injection strength parameter, b: recharge strength parameter
    """
    # read in time and dependent variable data
    [t_exact, x_exact] = [load_data()[2], load_data()[3]]

    # finding model constants in the formulation of the ODE using curve fitting
    # optimised parameters (pars) and covariance (pars_cov) between parameters
    pars,pars_cov = curve_fit(x_curve_fitting, t_exact, x_exact, pars_guess)
 
    return pars, pars_cov


# This function solves your ODE using Improved Euler for a future prediction with new q
def solve_ode_prediction(f, t0, t1, dt, xi, q, dqdt, a, b, c, x0):
    """ Solve the pressure prediction ODE model using the Improved Euler Method.
    Parameters:
    -----------
    f : callable
        Function that returns dxdt given variable and parameter inputs.
    t0 : float
        Initial time of solution.
    t1 : float
        Final time of solution.
    dt : float
        Time step length.
    xi : float
        Initial value of solution.
    a : float
        mass injection strength parameter.
    b : float
        recharge strength parameter.
    x0 : float
        Ambient value of solution.
    Returns:
    --------
    t : array-like
        Independent variable solution vector.
    x : array-like
        Dependent variable solution vector.
    Notes:
    ------
    Assume that ODE function f takes the following inputs, in order:
        1. independent variable
        2. dependent variable
        3. forcing term, q
        4. all other parameters
    """
    # finding the number of time steps
    tspan = t1 - t0
    n = int(tspan // dt)

    # initialising the time and solution vectors
    x = [xi]
    t = [t0]

    # using the improved euler method to solve the pressure ODE
    for i in range(n):
        f0 = f(t[i], x[i], q, dqdt,x0, a, b, c)
        f1 = f(t[i] + dt, x[i] + dt * f0, q, dqdt,x0, a, b, c)
        x.append(x[i] + dt * (f0 / 2 + f1 / 2))
        t.append(t[i] + dt)

    return np.array(t), np.array(x)


# This function plots your model over the data using your estimate for a and b
def plot_suitable():
    fig, (ax1, ax2) = plt.subplots(2, 1)

    # read in time and temperature data
    [t, x_exact] = [load_data()[2], load_data()[3]]

    # TYPE IN YOUR PARAMETER ESTIMATE FOR a AND b HERE (and c)
    pars = [0.01308,3.674e-9, 1]
  
    # solve ODE with estimated parameters and plot 
    x = x_curve_fitting(t, *pars)

    # convert back to MPa  and Days
    t /= 86400
    x_exact /= 1e6
    x /= 1e6

    ax1.plot(t, x_exact, 'k.', label='Observation')
    ax1.plot(t, x, 'r-', label='Curve Fitting Model')
    ax1.set_ylabel('Pressure (MPa)')
    ax1.set_xlabel('Time (day)')
    ax1.legend()

    # compute the model misfit and plot
    misfit = x
    for i in range(len(x)):
        misfit[i] = x_exact[i] - x[i]
    ax2.plot(t, misfit, 'x', label='misfit', color='r')
    ax2.set_ylabel('Pressure misfit (MPa)')
    ax2.set_xlabel('Time (day)')
    plt.axhline(y=0, color='k', linestyle='-')
    ax2.legend()

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()


# # This function plots your model over the data using your improved model after curve fitting.
def plot_improve():
    fig, (ax1, ax2) = plt.subplots(2, 1)

    # read in time and temperature data
    [t, x_exact] = [load_data()[2], load_data()[3]]

    # TYPE IN YOUR PARAMETER GUESS FOR a AND b HERE AS A START FOR OPTIMISATION
    pars_guess = [0.01308,3.674e-9, 1]
    
    # call to find out optimal parameters using guess as start
    pars, pars_cov = x_pars(pars_guess)

    # check new optimised parameters
    print ("Improved a and b and c")
    print (pars[0], pars[1], pars[2])
    

    # solve ODE with new parameters and plot 
    x = x_curve_fitting(t, *pars)

    t /= 86400
    x_exact /= 1e6
    x /= 1e6

    ax1.plot(t, x_exact, 'k.', label='Observation')
    ax1.plot(t, x, 'r-', label='Curve Fitting Model')
    ax1.set_ylabel('Pressure (MPa)')
    ax1.set_xlabel('Time (day)')
    ax1.legend()

    # compute the model misfit and plot
    misfit = x
    for i in range(len(x)):
        misfit[i] = x_exact[i] - x[i]
    ax2.plot(t, misfit, 'x', label='misfit', color='r')
    ax2.set_ylabel('Temp misfit (C)')
    ax2.set_xlabel('Time (sec)')
    plt.axhline(y=0, color='k', linestyle='-')
    ax2.legend()

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()


# This function plots your model against a benchmark analytic solution.
def plot_benchmark():
    """ Compare analytical and numerical solutions via plotting.

    Parameters:
    -----------
    none

    Returns:
    --------
    none

    """
    # values for benchmark solution
    t0 = 0
    t1 = 10
    dt = 0.1

    # model values for benchmark analytic solution
    a = 1
    b = 1
    c = 1

    # set ambient value to zero for benchmark analytic solution
    x0 = 0
    # set inital value to zero for benchmark analytic solution
    xi = 0

    # setup parameters array with constants
    pars = [x0, a, b, c]

    fig, plot = plt.subplots(nrows=1, ncols=3, figsize=(13, 5))

    # Solve ODE and plot
    t, x = solve_ode(ode_model, t0, t1, dt, xi, pars)
    plot[0].plot(t, x, "bx", label="Numerical Solution")
    plot[0].set_ylabel("Pressure [MPa]")
    plot[0].set_xlabel("t")
    plot[0].set_title("Benchmark")

    # Analytical Solution
    t = np.array(t)

    # TYPE IN YOUR ANALYTIC SOLUTION HERE
    x_analytical =  a * -1/b *(1 - np.exp(-b * t)) + x0

    plot[0].plot(t, x_analytical, "r-", label="Analytical Solution")
    plot[0].legend(loc=1)

    # Plot error
    x_error = []
    for i in range(1, len(x)):
        if (x[i] - x_analytical[i]) == 0:
            x_error.append(0)
            print("check line Error Analysis Plot section")
        else:
            x_error.append((np.abs(x[i] - x_analytical[i]) / np.abs(x_analytical[i])))
    plot[1].plot(t[1:], x_error, "k*")
    plot[1].set_ylabel("Relative Error Against Benchmark")
    plot[1].set_xlabel("t")
    plot[1].set_title("Error Analysis")
    plot[1].set_yscale("log")

    # Timestep convergence plot
    time_step = np.flip(np.linspace(1/5, 1, 13))
    for i in time_step:
        t, x = solve_ode(ode_model, t0, t1, i, x0, pars)
        plot[2].plot(1 / i, x[-1], "kx")

    plot[2].set_ylabel(f"Pressure(t = {10})")
    plot[2].set_xlabel("1/\u0394t")
    plot[2].set_title("Timestep Convergence")

    # plot spacings
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    plt.show()

def plot_x_forecast():
    ''' Plot the ODE LPM model over the given data plot with different q-value scenario for predictions.
    Use a curve fitting function to accurately define the optimum parameter values.
    Parameters:
    -----------
    none
    Returns:
    --------
    none
    '''
    # Read in time and dependent variable data 
    t_q, q, t, x_exact = load_data()
    q_interp = np.interp(t, t_q, q)
    last_q = q_interp[-1]

    # GUESS PARAMETERS HERE
    pars_guess = [0.01308,3.674e-9, 1]

    # Optimise parameters for model fit
    pars, pars_cov = x_pars(pars_guess)

    # Store optimal values for later use
    [a,b,c] = pars

    # Remember the last time
    t_end = t[-1]

    # Solve ODE and plot model
    x = x_curve_fitting(t, *pars)
    f, ax1 = plt.subplots()
    ax1.plot(t/86400, x_exact/1e6, 'r.', label='data')
    ax1.plot(t/86400, x/1e6, 'black', label='Model')

    # Create forecast time with 200 new time steps
    t1 = np.arange(20) * 86400 + t_end

    # Set initial and ambient values for forecast
    xi = x[-1] # Initial value of x is final value of model fit
    x0 = 0.01e6  # Ambient pressure value

    # injection rates for different stake holders
    q = [250,175,150]
    dt = t1[1] - t1[0]
    colors = ['purple', 'green', 'blue']

    for i , color in enumerate(colors):
        q_current = q[i]/3.6
        dqdt = (q_current - last_q) / dt
        x_result = solve_ode_prediction(ode_model, t1[0], t1[-1], dt, xi, q_current, dqdt, a,b,c,x0)[1]
        print(x_result[-1] / 1e6)
        ax1.plot(t1/86400, x_result/1e6, color, label=f'Prediction when q = {q[i]}')

    # Axis information
    ax1.set_title('Pressure Forecast')
    ax1.set_ylabel('Pressure (MPa)')
    ax1.set_xlabel('Time (day)')
    ax1.legend()
    plt.show()

# This function computes uncertainty in your model
def plot_x_uncertainty():
    """
    This function plots the uncertainty of the ODE model.
    """

    t_q, q, t, x_exact = load_data()
    q_interp = np.interp(t, t_q, q)
    last_q = q_interp[-1]

    # GUESS PARAMETERS HERE
    pars_guess = [0.01308,3.674e-9, 1]

    # Optimise parameters for model fit
    pars, pars_cov = x_pars(pars_guess)

    # Store optimal values for later use
    [a,b,c] = pars

    # Remember the last time
    t_end = t[-1]

    # Solve ODE and plot model
    x = x_curve_fitting(t, *pars)
    f, ax1 = plt.subplots()
    ax1.plot(t/86400, x_exact/1e6, 'r.', label='data')
    ax1.plot(t/86400, x/1e6, 'black', label='Model')

    # Create forecast time with 200 new time steps
    t1 = np.arange(20) * 86400 + t_end

    # Set initial and ambient values for forecast
    xi = x[-1] # Initial value of x is final value of model fit
    x0 = 0.01e6  # Ambient pressure value
    dt = t1[1] - t1[0]

    # Solve ODE prediction for scenario 1
    q1=250/3.6
    dqdt = (q1-last_q) / dt
    x1 = solve_ode_prediction(ode_model, t1[0], t1[-1], dt, xi, q1, dqdt, a, b, c, x0)[1]
    ax1.plot(t1/86400, x1/1e6, 'purple', label=f'Prediction when q = {q1*3.6}')

    # Solve ODE prediction for scenario 2
    q2=175/3.6
    dqdt = (q2 - last_q) / dt
    x2 = solve_ode_prediction(ode_model, t1[0], t1[-1], dt, xi, q2, dqdt, a, b, c, x0)[1]
    ax1.plot(t1/86400, x2/1e6, 'green', label=f'Prediction when q = {q2*3.6}')

    # Solve ODE prediction for scenario 3
    q3=150/3.6 
    dqdt = (q3- last_q) / dt
    x3 = solve_ode_prediction(ode_model, t1[0], t1[-1], dt, xi, q3, dqdt, a, b, c, x0)[1]
    ax1.plot(t1/86400, x3/1e6, 'blue', label=f'Prediction when q = {q3*3.6}')

    # Estimate the variability of parameter b
    # We are assuming that parameter b has the biggest source of error in the system (you could choose another parameter if you like)
    standard_dev=1e-7

    # using Normal function to generate 500 random samples from a Gaussian distribution
    samples = np.random.normal(b, standard_dev, 500)

    # initialise list to count parameters for histograms 
    b_list = []

    # loop to plot the different predictions with uncertainty
    for i in range(0,499): # 500 samples are 0 to 499
        # frequency distribution for histograms for parameters
        b_list.append(samples[i])

        # Solve model fit with uncertainty
        spars = [a, samples[i], c]
        x = x_curve_fitting(t, *spars)
        ax1.plot(t/86400, x/1e6, 'black', alpha=0.1, lw=0.5)

        # Solve ODE prediction for scenario 1 with uncertainty
        q1=250/3.6 # heat up again
        dqdt = (q1-last_q) / dt
        x1 = solve_ode_prediction(ode_model, t1[0], t1[-1], dt, xi, q1, dqdt, a, samples[i], c, x0)[1]
        ax1.plot(t1/86400, x1/1e6, 'purple', alpha=0.1, lw=0.5)

        # Solve ODE prediction for scenario 2 with uncertainty	
        q2=175/3.6 # keep q the same at zero
        dqdt = (q2 - last_q) / dt
        x2 = solve_ode_prediction(ode_model, t1[0], t1[-1], dt, xi, q2, dqdt, a, samples[i], c, x0)[1]
        ax1.plot(t1/86400, x2/1e6, 'green', alpha=0.1, lw=0.5)

        # Solve ODE prediction for scenario 3 with uncertainty
        q3=150/3.6 # extract at faster rate
        dqdt = (q3- last_q) / dt
        x3 = solve_ode_prediction(ode_model, t1[0], t1[-1], dt, xi, q3, dqdt, a, samples[i], c, x0)[1]
        ax1.plot(t1/86400, x3/1e6, 'blue', alpha=0.1, lw=0.5)

    ax1.set_title('Pressure Uncertainty Forecast')
    ax1.set_ylabel('Pressure (MPa)')
    ax1.set_xlabel('Time (day)')
    ax1.legend()

    # plotting the histograms
    figb, (ax2) = plt.subplots(1, 1)
    num_bins = 30
    ax2.hist(b_list, num_bins)
    ax2.set_title("Frequency Density plot for Parameter b", fontsize=9)
    ax2.set_xlabel('Parameter b', fontsize=9)
    ax2.set_ylabel('Frequency density', fontsize=9)
    a_yf5, a_yf95 = np.percentile(b_list, [5, 95])
    ax2.axvline(a_yf5, label='95% interval', color='r', linestyle='--')
    ax2.axvline(a_yf95, color='r', linestyle='--')
    ax2.legend(loc=0, fontsize=9)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()