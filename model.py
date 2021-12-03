from scipy.sparse.linalg import splu
import scipy.sparse as sp
import numpy as np
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy as sc
from scipy.sparse import diags



def logdet(A):
    
    # compute log determinant of a sparse matrix A
    if not sp.issparse(A):
        LU = splu( sp.csc_matrix(A) )
    else:   
        LU = splu(A)
        
    #scipy.sparse.csc_matrix.diagonal:
    logdetl = np.sum(np.log(LU.L.diagonal()))
    logdetu = np.sum(np.log(LU.U.diagonal()))
    
    return logdetu + logdetl


def solve_tridiag(A,B):
    # solve x = A\d for a tridiagonal matrix A; Thomas algorithm
    
    N = A.shape[0]
    x = np.zeros((N,1))
    
    # check A is an array; 
    # get the diagonals
    a = np.zeros(N,)
    a[1:] = np.copy( np.diag( A, -1) )
    b = np.copy( np.diag( A ) )
    c = np.zeros(N,)
    c[:-1] = np.copy( np.diag( A, 1) )
    d = np.copy(B)
    
    for i in range(1,N):
    
        w = a[i]/b[i-1]
        b[i] -= w*c[i-1]
        d[i] -= w*d[i-1]
    
    x[-1] = d[-1]/b[-1]
    
    for i in range(N-2,-1,-1):
        x[i] = (d[i] - c[i]*x[i+1]) / b[i]
    
    return x


def calcC_useSparse_permute(N,M,sgma,one_block=False,return_sparse=False):
    # compute inverse prior covariance and its logdet, use sparse matrices and block operations
    # assumes that vector of weights is in format [w[1,t=0]...w[1,t=T],w[2,t=0]...w[N,t=T]]'
    # one_block=True means that instead of returning a full matrix [N*M,N*M] we only return one block [N,N]
    
    sginv = sgma**(-2) * sp.eye(N)
    
    D = sp.eye(N) + sp.diags(-1 * np.ones(N-1,), -1)
        
    Cm1_block = D.transpose().dot(sginv).dot(D) 
    
    logdetCm1_block = logdet(Cm1_block)
    
    if one_block:
        if return_sparse:
            return Cm1_block, logdetCm1_block
        else:
            return Cm1_block.todense(), logdetCm1_block
    else:
        logdetCm1 = logdetCm1_block * M
        if return_sparse:
            Cm1 = sp.block_diag( [Cm1_block]*M )
            return Cm1, logdetCm1
        else:
            Cm1_block = Cm1_block.todense()
            Cm1 = sc.linalg.block_diag( *([Cm1_block]*M) )
            return Cm1, logdetCm1
        

def calcE_ind_permute(this_cov, this_s, trial_ids, data, params):
    # compute log marginal likelihood E using a subset of trials (ind) 
    # preprequisites: x has to be in format [x[0,t=0]...x[0,t=T],x[1,t=0]...x[N,t=T]]' 
    
    xz = data['xz']
    xx = data['xx']
    z = data['z']
    N = params['N']
    M = params['krnlen']
        
    if len(trial_ids)==0:
        return 0
    
    # subselect rows/columns in the C, not C^-1
    Cm1_b,_ = calcC_useSparse_permute(N, M, this_cov, one_block=True)
    C_b = np.linalg.inv(Cm1_b)
    Csub_b = C_b[np.ix_(trial_ids, trial_ids)]
    Cm1sub_b = np.linalg.inv(Csub_b)
    Cm1Sub = sc.linalg.block_diag( *([Cm1sub_b]*M) )
    Cm1SubLogdet = logdet(Cm1Sub)*M

    #
    logdetI = logdet((2*np.pi*this_s**2)*np.eye( len(trial_ids)))
    
    # 
    row_ids = trial_ids[:,np.newaxis] + N*np.arange(M,)[np.newaxis,:]
    row_ids = np.sort( row_ids.ravel() )   
    Asub = xx[np.ix_(row_ids, row_ids)] + (this_s**2)*Cm1Sub
    Bsub = xz[trial_ids,:].T.ravel()# select trials, then ravel row by row to get [-x1-,-x2-].T
    Am1BTBsub = solve_tridiag(Asub,Bsub).T.dot(Bsub)
    SigPostSub = (1/this_s**2) * Asub
    
    #
    zsub = z[trial_ids]
    zTzsub = (0.5/(this_s**2))*zsub.T.dot(zsub) 
    
    #
    Esub = -0.5*logdet(SigPostSub) - 0.5*logdetI + 0.5*Cm1SubLogdet + (0.5/(this_s**2))*Am1BTBsub - zTzsub
    
    return Esub



def plotPrior(f_prior,particles):
    
    sgma_grid = np.linspace(-3,2,15)
    s_grid = np.linspace(-3,2,15)

    prval = np.zeros((15,15))

    for i,sgma in enumerate(sgma_grid):
        for j,s in enumerate(s_grid):
            prval[i,j] = f_prior( sgma, s )

    xx,yy = np.meshgrid( s_grid, sgma_grid)
    plt.figure()
    plt.contour(xx,yy,prval)
    plt.xlabel('log(s)')
    plt.ylabel('log(sgma)')



def partguide_exp(data, params, params_pf, f_prior, do_plot=False):
    # iteratively fits parameters of a random walk process that generates data
    
    # steps here:
    # setup q (proposal distribution) parameters
    # loop over progressively larger data partitions (1:p)
    # * weight every particle by exp(E[next data] - E[this data])
    # sample ancestors (duplicate particles with high weight remove with low weight) - sample from polynomial distribution
    # propose noisy values around resampled points using q distribution
    # ** compute ratio of pi-prime (proposal) and pi (current) particle, decide which one to keep
    # next data iteration
    
    # * data next / data this ; ** particle proposal / particle current
    
    N = params['N']
    
    P = params_pf['P']
    ind_perm = params_pf['ind_perm']
    N_part = params_pf['N_part']
    particles = params_pf['particles']
    maxinds = np.arange(P+1)*N//P
    if 'proposal_var' in params_pf:
        prop_var = params_pf['proposal_var']
    else:
        prop_var = 0.01

    proposal_cov = [[prop_var, 0],[0, prop_var]];
    rng = np.random.default_rng()
        
        
    # # # # plot-1: prior and first sample of particles
    if do_plot:
        plt.figure()
        plotPrior(f_prior, particles)
        plt.scatter( np.squeeze(particles[0,:,0]), np.squeeze(particles[1,:,0]) )
        ax = plt.gca()

        
    for p in range(P):

        print(f'data partition {p+1} of {P}...') 

        # "future" ids
        idf = np.sort( ind_perm[:maxinds[p+1]] )
        # "current" ids
        idc = np.sort( ind_perm[:maxinds[p]] )

        weights = np.ones((N_part,1))/N_part
        Enext = np.zeros((N_part,1))
        Ethis = np.zeros((N_part,1))

        for i in range(N_part):
            this_cov = 10**particles[0,i,p]
            this_s = 10**particles[1,i,p]

            Enext[i] = calcE_ind_permute(this_cov, this_s, idf, data, params) + np.log(f_prior( np.log10(this_cov), np.log10(this_s) ))
            Ethis[i] = calcE_ind_permute(this_cov, this_s, idc, data, params) + np.log(f_prior( np.log10(this_cov), np.log10(this_s) ))
            
        weights = np.exp( Enext-Ethis )
        weights[np.isnan(weights)] = 0
        weights = weights/sum(weights)
        
        # # # # plot-2: posterior on the grid
        #if do_plot:
        #    plt.figure()
        #    calcPostGrid(data, params, idf, f_prior)    

            
            
        # [2] ancestor sampling: duplicate particles with higher weight

        n_resample = rng.multinomial(len(weights),weights,size=1)
        n_resample = n_resample[0]
        part_resampled = np.zeros((len(weights),),dtype='int')

        k=0
        for i in range(len(weights)):
            this_n = n_resample[i]
            part_resampled[k:k+this_n] = i
            k += this_n

        particles[:,:,p] = particles[:, part_resampled ,p]
        weights = np.ones((N_part,1))/N_part

        # # # # plot-3 (on top of 1): - show high weight particles
        if do_plot:
            ax.scatter( np.squeeze(particles[0,:,p]), np.squeeze(particles[1,:,p]), s=100, marker='+' )
            plt.title('prior(contour),part0:blue,pluses:resampled')
        
        
        
        # [3] propose particles around the resampled ones using q distribution; compute posterior for the original and proposed
        prop_part = np.zeros((2,N_part))
        for i in range(N_part):
            prop_part[:,i] = mvn.rvs(particles[:,i,p],proposal_cov)

        # using the "future" data, compute posterior of the proposed and current particle
        Eprop = np.zeros((N_part,1))
        Ecurr = np.zeros((N_part,1))
        alpha = np.zeros((N_part,1))
        
        for i in range(N_part):

            prop_cov = 10**prop_part[0,i]
            prop_s = 10**prop_part[1,i]

            Eprop[i] = calcE_ind_permute(prop_cov, prop_s, idf, data, params)  + np.log( f_prior( np.log10(prop_cov), np.log10(prop_s) ) )

            this_cov = 10**particles[0,i,p]
            this_s = 10**particles[1,i,p]    

            Ecurr[i] = calcE_ind_permute(this_cov, this_s, idf, data, params)  + np.log( f_prior( np.log10(this_cov), np.log10(this_s) ) )  
            
            if Eprop[i]-Ecurr[i]>0:
                alpha[i] = 1.0
            else:
                alpha[i] = np.exp( Eprop[i] - Ecurr[i] ) 
            #alpha[i] = min(1, np.exp( Eprop[i] - Ecurr[i] ) )

        # use proposed or keep original? randomized criterion here (optional)   
        decision = (alpha > np.random.rand(N_part,1)).T

        # set particle values for the next 
        particles[:,:,p+1] = particles[:,:,p]*(1-decision) + prop_part*decision
        
        
        
        # # # # plot-4: show previous particle, show all proposed, show all selected for the next iteration
        if do_plot:
            plt.figure()
            plotPrior(f_prior, particles)
            plt.scatter( np.squeeze(particles[0,:,p]), np.squeeze(particles[1,:,p]), s=100 )
            plt.scatter( prop_part[0,:], prop_part[1,:] , s=100, marker='o', facecolors='none', edgecolors='k')
            plt.scatter( np.squeeze(particles[0,:,p+1]), np.squeeze(particles[1,:,p+1]) , s=50, marker='x')
            plt.title('blue:ancestor,O:proposed,X:accepted')
    
    return particles