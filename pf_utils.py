import numpy as np
from scipy.sparse import diags

def simpfdata( N=100, sgma=0.1, s=0.1, krnlen=1, randomSeed=0):
    # simulate data for particle filter
    # inputs:
    # N - number of trials
    # sgma - volatility (std)
    # s - measurement noise (std)
    # krnlen - length of response kernel, timebins
    
    rng = np.random.RandomState(randomSeed)

    k_all = np.zeros((N,krnlen)) # dynamic response kernel, i.e. state
    x = rng.randn(N,krnlen) # inputs
    z = np.zeros((N,1))
    
    for i in range(N-1):
        k_this = k_all[i]
        k_all[i+1] = k_this + sgma*rng.randn(krnlen,)
        
        z[i] = x[i]@(k_this.T) + s*np.random.randn()
    z[-1] = x[-1]@(k_this.T) + s*np.random.randn()


    # useful stats
    
    #xxT = np.diag( np.reshape(x*x,(-1,)) ) # 1d case only
    #xxT = np.diag( np.reshape(x.ravel()*x.ravel(),(-1,)) ) # 1d case only
    
    xprod = np.einsum('ij,ik->ijk',x,x)
    xxT = myblk_diags(xprod) # output is sparse

    B = x*z
    
    simdata = {'k':k_all, 'x':x, 'z':z, 'xz':B, 'xx':xxT}

    return simdata


def myblk_diags(A):
    # input A [N,K,K], put each len(N) entry of A[:,i,j] as
    # the diagonal of an (N*K x N*K) matrix. 
    # Borrowed from N.Roy's package psytrack 
    # used e.g. in creating xxT matrix when the long weight vector is horzstack of each of its dimensions' time-vectors 
    
    # Retrieve shape of given matrix
    N, K, _ = np.shape(A)

    # Will need (2K-1) diagonals, with the longest N*K long
    d = np.zeros((2 * K - 1, N * K))

    # Need to keep track of each diagonal's offset in the final
    # matrix : (0,1,...,K-1,-K+1,...,-1)
    offsets = np.hstack((np.arange(K), np.arange(-K + 1, 0))) * N

    # Retrieve diagonal values from A to fill in d
    for i in range(K):
        for j in range(K):
            m = np.min([i, j])
            d[j - i, m * N:(m + 1) * N] = A[:, i, j]

    # After diagonals are constructed, use sparse function diags() to make
    # matrix, then blow up to full size
    return diags(d, offsets, shape=(N * K, N * K), format='csc')
