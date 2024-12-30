#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def signal_predection_Laplacian(D, psi_tilde, tau = 10): #LSP
    """
    Input:
    D: Dirac Operator
    psi_tilde: Noisy signal
    tau: Regularisation parameter

    
    
    Output:
    psi_hat: Predicted signal
    
    """
    
    
    #GB-Laplacian matrix
    I = np.identity(len(D))
    L = matrix_power(D,2)
    psi_hat = np.linalg.inv(I + tau * L).dot(psi_tilde)
    return psi_hat


def signal_predection_Dirac(D, psi_tilde, tau = 10, sigma = 1/2, delta = 1e-4, T = 10): #DSP
    """
    Input:
    D: Dirac Operator
    psi_tilde: Noisy signal
    tau: Regularisation parameter
    sigma: Initial step size
    delta: Tolerance on convergence
    T: Minimum iterations
    
    
    Output:
    E: Predicted corresponed eigenvalue
    psi_hat: Predicted signal
    
    """
    
    def func(D,psi_hat,psi_tilde,tau,E): #Evaluation of the function 
        I = np.identity(len(D))
        Norm = np.linalg.norm(psi_hat - psi_tilde,2)**2
        f = Norm + tau * psi_hat.transpose().dot(matrix_power(D - E * I,2).dot(psi_hat))
        return f

    psi_hat = psi_tilde
    psi_second_to_last = np.zeros(len(psi_tilde))
    E = (psi_hat.transpose().dot((D).dot(psi_hat)))/(np.linalg.norm(psi_hat,2)**2)
    I = np.identity(len(D))
    while True:
        A = D - E * I
        psi_hat_new = (np.linalg.inv(I + tau * np.conjugate(A).dot(A))).dot(psi_tilde)
        E_new = (1-sigma) * E + sigma * (psi_hat.transpose().dot((D).dot(psi_hat)))/(np.linalg.norm(psi_hat,2)**2)
        E_exp = psi_hat_new.T.dot(D).dot(psi_hat_new) / (np.linalg.norm(psi_hat_new, 2 ) ** 2)
        #Armijo rule of choosing step size to guarantee convergence.
        if func(D,psi_hat,psi_tilde,tau,E) > func(D,psi_hat,psi_tilde,tau,E): 
            sigma /= 2 
            
        else:
            psi_hat = psi_hat_new
            E = E_new
            T -= 1
        if T < 0 and (abs(E_exp - E) < delta or sigma < delta): #If converges
            break
    return E, psi_hat


#DESP_L (based on Loss)
def signal_predection_Dirac_Equation(D, psi_tilde, Gamma, ub_m = 2, lb_m = -2, tau = 10, sigma = 1/8, delta = 1e-3, T = 10, n = 41):
    
    """
    Input:
    D: Dirac Operator
    psi_tilde: Noisy signal
    Gamma: Gamma matrix
    tau: Regularisation parameter
    sigma: Initial step size
    delta: Tolerance on convergence
    T: Minimum iteration
    n: Number of sampled m
    
    Output:
    E: Predicted energy
    m: Predicted mass
    psi_hat: Predicted signal
    
    """
    
    def func(D,psi_hat,psi_tilde,tau,E): #Evaluation of the function 
        I = np.identity(len(D))
        Norm = np.linalg.norm(psi_hat - psi_tilde,2)**2
        f = Norm + tau * psi_hat.transpose().dot(matrix_power(D - E * I,2).dot(psi_hat))
        return f
    
 #This can be changed to larger range
    m_list = np.linspace(lb_m,ub_m,n)
    
    E_over_m = [] #Store all the results of psi_hat and error based on different m.
    psi_hat_over_m = []
    Loss_over_m = []
    
    for m in m_list:
        E,psi_hat = signal_predection_Dirac(D + m * Gamma, psi_tilde, tau, sigma, delta, T)
        E_over_m.append(E)
        psi_hat_over_m.append(psi_hat)
        Loss_over_m.append(func(D + m * Gamma, psi_hat, psi_tilde, tau, E))
    
    #We want to output the psi_hat that has lowest loss value.
    ind = Loss_over_m.index(min(Loss_over_m))
    E = E_over_m[ind]
    m = m_list[ind]
    psi_hat = psi_hat_over_m[ind]
    
    return E, m, psi_hat


#DESP_S (based on dispersion)
def signal_predection_Dirac_Equation_S(D, psi_tilde, Gamma, ub_m = 5, lb_m = -5, tau = 10, sigma = 1/8, delta = 1e-4, T = 10, n = 41):
    
    """
    Input:
    D: Dirac Operator
    psi_tilde: Noisy signal
    Gamma: Gamma matrix
    tau: Regularisation parameter
    sigma: Initial step size
    delta: Tolerance on convergence
    T: Minimum iteration
    n: Number of sampled m
    
    Output:
    E: Predicted energy
    m: Predicted mass
    psi_hat: Predicted signal
    
    """
    
 #This can be changed to larger range
    m_list = np.linspace(lb_m,ub_m,n)
    
    E_over_m = [] #Store all the results of psi_hat and error based on different m.
    psi_hat_over_m = []
    Disp_over_m = []
    
    for m in m_list:
        E,psi_hat = signal_predection_Dirac(D + m * Gamma, psi_tilde, tau, sigma, delta, T)
        E_over_m.append(E)
        psi_hat_over_m.append(psi_hat)
        Disp_over_m.append(RDRE(D,psi_hat,m, Gamma, E))
    
    #We want to output the psi_hat that has lowest loss.
    ind = Disp_over_m.index(min(Disp_over_m))
    E = E_over_m[ind]
    m = m_list[ind]
    psi_hat = psi_hat_over_m[ind]
    
    return E, m, psi_hat

#IDESP
def IDESP(D, Gamma, psi_true, psi_tilde, method, tau = 15, sigma = 1/2, delta = 1e-3, T = 10):
    """
    
    Input:
    D: Dirac operator
    Gamma: Gamma matrix
    psi_true: True signal
    psi_tilde: Noisy signal
    method: Which algorithm to use
    
    
    Output:
    psi_hat_res: Prediction of signal (with combination)
    psi_hat_J: List of psi_hat_res over J
    Error_J: Tendency of errors to the true signal over the number of iterations
    c_V_True: Parameter
    c_V_J: Parameter
    """
    
    
    
    c_V_True = np.linalg.norm(psi_true - psi_tilde,2)
    if method == signal_predection_Laplacian:
        psi_hat_res = signal_predection_Laplacian(D, psi_tilde, tau)
        psi_hat_J = [psi_hat_res]
        Error_J = [np.linalg.norm(psi_hat_res - psi_tilde,2)]
        c_V_J = [np.linalg.norm(psi_hat_res-psi_tilde, 2)/ np.linalg.norm(psi_hat_res, 2)]
        return c_V_True, psi_hat_res, psi_hat_J,  Error_J, c_V_J
    
    if method == signal_predection_Dirac:
        psi_hat_res = signal_predection_Dirac(D, psi_tilde)[1]
        m = 0
    elif method == signal_predection_Dirac_Equation:
        E, m, psi_hat_res = signal_predection_Dirac_Equation(D, psi_tilde, Gamma)
    else:
        E, m, psi_hat_res = signal_predection_Dirac_Equation_S(D, psi_tilde, Gamma)
    
    psi_hat_J = [psi_hat_res.copy()]
    Error_J = [np.linalg.norm(psi_hat_res-psi_true, 2)/ np.linalg.norm(psi_true, 2)]
    c_V_J = [np.linalg.norm(psi_hat_res-psi_tilde, 2)/ np.linalg.norm(psi_hat_res, 2)]
    c_V_Difference = abs(c_V_True - c_V_J[-1])
    
    while True:
        psi_hat = signal_predection_Dirac(D + m * Gamma, psi_tilde - psi_hat_res)[-1]
        psi_hat_res += psi_hat
        c_V_J.append(np.linalg.norm(psi_hat_res-psi_tilde, 2)/ np.linalg.norm(psi_hat_res, 2))
        c_V_Difference_new = abs(c_V_True - c_V_J[-1])
        if c_V_Difference_new > c_V_Difference:
            psi_hat_res -= psi_hat
            break
        else:
            psi_hat_J.append(psi_hat_res.copy())
            Error_J.append(np.linalg.norm(psi_hat_res-psi_true, 2)/ np.linalg.norm(psi_true, 2))
            c_V_Difference = c_V_Difference_new

    return c_V_True, psi_hat_res, psi_hat_J, Error_J, c_V_J





#Generate synthetic signals whose edges and nodes share the same noise scale alpha. 
def generate_signals_eigenvalues_F(D,alpha,position):
    """
    Input:
    D: Dirac Operator
    alpha: Error size
    position: Index of eigenvector (sorted by eigenvalue)
    
    
    Output:
    Lambda: Eigenvalue
    psi_true: Unit true signal (norm = 1)
    psi_tilde: Noisy signal
    
    """
    noise = np.random.normal(0,1,D.shape[1])
    noise = ((D).dot(np.linalg.pinv(D))).dot(noise.real)
    noise = alpha*noise/np.sqrt(np.linalg.matrix_rank(D))
    Lambdas, phis = np.linalg.eigh(D)
    Lambdas_real = np.real_if_close(Lambdas)
    try:
        Lambda = Lambdas_real[position]
        psi_true = np.real(phis[:, position])
        
        psi_tilde = psi_true + noise
        return Lambda, psi_true, psi_tilde
    except:
        print ("Position should between 0 to " + str(len(phis)-1))
        
        
#Generate synthetic signals whose edges have noise scale alpha_1 and nodes have noise scale alpha_2.
def generate_signals_eigenvalues_FF(D, N, L, alpha_1, alpha_2,position):
    """
    Input:
    D: Dirac Operator
    alpha_1: Error scale on nodes
    alpha_2: Error scale on links
    position: Index of eigenvector (sorted by eigenvalue)
    
    
    Output:
    Lambda: Eigenvalue
    psi_true: Unit true signal (norm = 1)
    psi_tilde: Noisy signal
    
    """
    
    alpha = np.concatenate((np.ones(N) * alpha_1, np.ones(L) * alpha_2))
    noise = np.random.normal(0,1,D.shape[1])
    noise = ((D).dot(np.linalg.pinv(D))).dot(noise.real)
    noise = alpha*noise/np.sqrt(np.linalg.matrix_rank(D))
    Lambdas, phis = np.linalg.eigh(D)
    Lambdas_real = np.real_if_close(Lambdas)
    try:
        Lambda = Lambdas_real[position]
        psi_true = np.real(phis[:, position])
        
        psi_tilde = psi_true + noise
        return Lambda, psi_true, psi_tilde
    except:
        print ("Position should between 0 to " + str(len(phis)-1))
        
#Generate noisy signal for real signals depends on node/edge error scale alpha.
def noisy_signal(D,psi_true, alpha):
    """
    Input:
    D: Dirac Operator
    psi_true: True signal
    alpha: Error scale
    
    
    Output:
    psi_tilde: Noisy signal
    
    """
    w,v = np.linalg.eigh(D)
    v_nonzero = v[:, (~np.isclose(w, 0))]
    w_nonzero = w[(~np.isclose(w, 0))]
    k = np.linalg.norm(psi_true, 2)
    noise = np.random.normal(0,k,psi_true.shape[0])/np.sqrt(len(w_nonzero)) #Remove the harmonic components
    noise = ((v_nonzero).dot(np.transpose(v_nonzero))).dot(noise)
    psi_tilde = psi_true +  alpha * noise
    return psi_tilde


#Calculate RDRE.
def RDRE(D,psi_hat,m, Gamma, E):
    Lambda_square = psi_hat.transpose().dot(D.dot(D)).dot(psi_hat) / (np.linalg.norm(psi_hat, 2) ** 2)
    E = psi_hat.T.dot(D + m * Gamma).dot(psi_hat) / (np.linalg.norm(psi_hat, 2 ) ** 2)
    return abs(E ** 2 - Lambda_square - m ** 2 )

#Calculate Loss.
def loss(D,m, Gamma,E,psi_tilde,psi_hat,tau):
    I = np.identity(len(D))
    Norm = np.linalg.norm(psi_hat - psi_tilde,2)**2
    f = Norm + tau * psi_hat.transpose().dot(matrix_power(D + m * Gamma - E * I,2).dot(psi_hat))
    return f



#Calculate Lm and Sm.
def compare_error(D, psi_true, psi_tilde, Gamma, ub_m, lb_m, tau = 10, sigma = 1/2, delta = 1e-8, T = 10, n = 101):
    
    """
    Input:
    D: Dirac operator
    psi_true: True signal
    psi_tilde: Noisy signal
    Gamma: Gamma matrix
    ub_m: Upper bound for m
    lb_m: Lower bound for m
    tau: Regularisation parameter
    sigma: Initial step size
    delta: Tolerance on convergence
    T: Minimum iteration
    n: Number of sampled m
    
    Output:
    
    
    True_error: List of errors || psi_hat - psi_true || over m
    Loss: List of losses L over m
    dispersion: List of dispersion abs(E ** 2 - Lambda_square - m ** 2 ) over m
    m_list: Lists of sampled m
    """
    
    Loss = []
    Dispersion = []
    True_error = []
    m_list = np.linspace(ub_m,lb_m,n)
    psi_true /= np.linalg.norm(psi_true,2) #Normalize it.
    for m in m_list:
        E,psi_hat = signal_predection_Dirac(D + m * Gamma, psi_tilde, tau, sigma, delta, T)
        True_error.append(np.linalg.norm(psi_hat - psi_true,2))
        Loss.append(loss(D,m, Gamma,E,psi_tilde,psi_hat,tau))
        Dispersion.append(RDRE(D,psi_hat,m, Gamma, E))
    return np.array(True_error), np.array(Loss), np.array(Dispersion), m_list

#Plot for the std and mean
def plot_mean_and_CI(m_list, mean, lb, ub, color_mean=None, color_shading=None, label = None):
    plt.fill_between(m_list, ub, lb, color=color_shading, alpha=.2)
    plt.plot(m_list, mean, color = color_mean, label = label)
class LegendObject(object):
    def __init__(self, facecolor='red', edgecolor='white', dashed=False):
        self.facecolor = facecolor
        self.edgecolor = edgecolor
        self.dashed = dashed
 
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpatches.Rectangle(
            # create a rectangle that is filled with color
            [x0, y0], width, height, facecolor=self.facecolor,
            # and whose edges are the faded color
            edgecolor=self.edgecolor, lw=3)
        handlebox.add_artist(patch)
 
        # if we're creating the legend for a dashed line,
        # manually add the dash in to our rectangle
        if self.dashed:
            patch1 = mpatches.Rectangle(
                [x0 + 2*width/5, y0], width/5, height, facecolor=self.edgecolor,
                transform=handlebox.get_transform())
            handlebox.add_artist(patch1)
        return patch
    