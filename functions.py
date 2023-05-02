import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import sys


from parameters import params

class QuantumWell():
    C = 1.602176565e-19 #C #Charge of electron
    hb = 1.054571817e-34 #Js #Reduced Planck's constant
    m0 = 9.10938291e-31 #kg #Rest mass of electron
    def __init__(self, W, well='GaAs', barrier='AlAs', phase='cub', plot=True):

        self.setup_parameters(W, well, barrier, phase, plot)

        
        #self.normalize(self.bandEc, self.Ec, self.kc, self.thetac)
        
    def setup_parameters(self, W_L, well, barrier, phase, plot):
        self.plot = plot
        self.well = well
        self.barrier = barrier
        self.phase = phase
        self.m = {'c':'me', 'v':'mh'}
        self.E = {}
        self.k = {}
        self.t = {}
        self.dE = []

        self.dtheta = 0.0001
        self.EgW = params[self.well][self.phase]['Eg']
        V0c = 1
        V0v = 0.3465
        self.V0c = V0c * self.C
        self.V0v = V0v * self.C
        self.V0 = {'c': V0c, 'v': V0v}
        self.W_L = self.check_W_L(W_L)
        self.setup(self.W_L, self.plot)
        
    def check_W_L(self, W_L):

        '''
        Converts W_L to numpy array in meters
        
        W_L: int or float in nm
        '''

        try:
            if W_L.shape[0] > 1:
                self.plot= False
        except:
            try:
                W_L = np.arange(W_L, W_L + 0.99, 1.0)
                
            except:
                print('Wrong format for W_L')
                sys.exit()
        W_L = np.round(W_L, 4)

        return W_L
        
    def setup(self, W_L, plot): 
        for W in W_L:
            #Prepare containers for energies, k, and theta for each W_L, for each band.
            self.E[str(W)] = {}
            self.E[str(W)]['c'] = []
            self.E[str(W)]['v'] = []
            self.k[str(W)] = {}
            self.k[str(W)]['c'] = []
            self.k[str(W)]['v'] = []
            self.t[str(W)] = {}
            self.t[str(W)]['c'] = []
            self.t[str(W)]['v'] = []
            
            Ec, kc, tc = self.energy_calculator(self.V0['c'] * self.C, W * 1e-9, 'c' , plot)
            Ev, kv, tv = self.energy_calculator(self.V0['v'] * self.C, W * 1e-9, 'v', plot)
            
            self.E[str(W)]['c'] = Ec
            self.E[str(W)]['v'] = Ev 
            self.k[str(W)]['c'] = kc
            self.k[str(W)]['v'] = kv
            self.t[str(W)]['c'] = tc
            self.t[str(W)]['v'] = tv

            self.dE.append(Ec[0] + self.EgW + Ev[0])

        
        #if self.W_L.shape[0] > 1:
             #fig, ax = plt.subplots(dpi=600)
             #plt.plot(self.W_L, dE)
    def plot_W_L(self):
        if self.W_L.shape[0] > 1:
            fig, ax = plt.subplots(dpi=500)
            ax.plot(self.W_L, self.dE)
            ax.set_xlabel(r'$L_W$')
            ax.set_ylabel(r'Emission')
       
    def energy_calculator(self, V0, W_L, band, plot):
        C = 1.602176565e-19 #C #Charge of electron
        hb = 1.054571817e-34 #Js #Reduced Planck's constant
        m0 = 9.10938291e-31 #kg #Rest mass of electron

        mW = params[self.well][self.phase][self.m[band]]
        mB = params[self.barrier][self.phase][self.m[band]]
        theta0 = np.sqrt(m0 * mW * V0 * W_L ** 2 / (2 * hb ** 2) )
        NoOS = np.ceil(theta0/(np.pi/2));
        
        theta = np.arange(self.dtheta, theta0, self.dtheta);
        
        rightHandSide = np.sqrt( mW / mB * (theta0 ** 2 / theta ** 2 - 1) )
        leftHandSideEven = np.tan(theta)
        leftHandSideOdd = -1/np.tan(theta)

        theta = self.intersection_finder(NoOS, theta, mW, mB, theta0, rightHandSide, 
                                     leftHandSideEven, leftHandSideOdd, plot)
    
        theta = np.sort(theta)
        k = 2 * theta / W_L
        E = hb ** 2 * k ** 2 / (2*m0*mW*C)
        return E, k, theta
    
    def normalize(self, band, W=None, plot=False):
        '''
        band: str
        a: int or float in nm
        '''
        if band !='c' and band!='v':
            print('Incorrect band \'c\' or \'v\'')
            return
        if W == None:
            W_L = self.W_L[0]
        else:
            if str(float(W)) not in self.E.keys():
                self.setup(self.check_W_L(W), plot)
            W_L = W
        V0 = self.V0[band]
        mW = params[self.well][self.phase][self.m[band]]
        mB = params[self.barrier][self.phase][self.m[band]]
        
        k = self.k[str(float(W_L))][band]
        theta = self.t[str(float(W_L))][band]
        E = self.E[str(float(W_L))][band]

        A = np.zeros(k.shape[0])
        B = np.zeros(k.shape[0])

        k=k*1e-9 #To nm^-1
        
        V0 = V0
        kappa = np.zeros(k.shape[0])
        kappa[::2] = k[::2]*mB/mW*np.tan(theta[::2])
        kappa[1::2] = -k[1::2]*mB/mW/np.tan(theta[1::2])
        #Ec 2.111 Quantum wells, wires and dots Harrison
        A[::2] = 1/np.sqrt(W_L/2 + np.sin(k[::2]*W_L)/(2 * k[::2]) + np.cos(k[::2]*W_L/2) ** 2 / kappa[::2]) 
        B[::2] = A[::2] * np.exp(kappa[::2] * W_L / 2) * np.cos(k[::2] * W_L / 2)
        #Ec 2.112 Quantum wells, wires and dots Harrison
        A[1::2] = 1/np.sqrt(W_L/2 - np.sin(k[1::2]*W_L)/(2 * k[1::2]) + np.sin(k[1::2]*W_L/2) ** 2 / kappa[1::2]) 
        B[1::2] = A[1::2] * np.exp(kappa[1::2] * W_L / 2) * np.sin(k[1::2] * W_L / 2)
        W_Ls = W_L/1000 #W_L step
        W_Ll = []
        W_Lrange = 3*W_L
        for _ in k:
            W_Ll.append(np.arange(-W_Lrange, W_Lrange + W_Ls, W_Ls))
        W_Lr = np.array(W_Ll)
        W_Ln = int((W_Lrange - W_L/2)/W_Ls) #index of -W_L
        W_Lp = int((W_Lrange + W_L/2)/W_Ls)
        
        psi = np.zeros((k.shape[0],W_Lr.shape[1]))
        psi[::2,:W_Ln] = (np.exp(W_Lr[::2].T[:W_Ln,:]*(kappa[::2]))*B[::2]).T
        psi[::2,W_Ln:W_Lp] = (np.cos(W_Lr[::2].T[W_Ln:W_Lp,:]*(k[::2]))*A[::2]).T
        psi[::2,W_Lp:] = (np.exp(-W_Lr[::2].T[W_Lp:,:]*(kappa[::2]))*B[::2]).T
        
        psi[1::2,:W_Ln] = (np.exp(W_Lr[1::2].T[:W_Ln,:]*(kappa[1::2]))*(-B[1::2])).T
        psi[1::2,W_Ln:W_Lp] = (np.sin(W_Lr[1::2].T[W_Ln:W_Lp,:]*(k[1::2]))*A[1::2]).T
        psi[1::2,W_Lp:] = (np.exp(-W_Lr[1::2].T[W_Lp:,:]*(kappa[1::2]))*B[1::2]).T
        
        normalization = (integrate.simpson(psi ** 2, dx=W_Ls)) # psi - nergies for the integration and confirm normalization

        print(normalization)
        
        fig, ax = plt.subplots(dpi=500)
        ax.plot(W_Lr.T, psi.T + E, linewidth=0.5)
        wellx = [-W_Lrange, -W_L/2, -W_L/2, W_L/2, W_L/2, W_Lrange]
        welly = [V0, V0, 0,0, V0, V0]
        ax.set_ylabel(r'$E$')
        ax.set_xlabel(r'$L$')
        plt.plot(wellx, welly, color='black', linewidth=1)
        energyx = [-W_Lrange, W_Lrange]
        Ey = np.array([E, E])
        plt.gca().set_prop_cycle(None)
        plt.plot(energyx, Ey, '-.', linewidth=0.5)
        plt.show()

    def intersection_finder(self, NoOS, theta, mW, mB, theta0, rightHandSide, leftHandSideEven, leftHandSideOdd, plot):
        solutions = np.zeros(int(NoOS))
        zero = rightHandSide - leftHandSideEven
        zeros = (np.diff(np.sign(zero)) != 0)
        zeros_index = np.nonzero(zeros)
        theta_zero = theta[zeros_index]
        leftHandSideEven[zeros_index] = None
        
        x = np.tan(theta_zero) - np.sqrt(mW / mB * (theta0 ** 2 / theta_zero ** 2 - 1) )
        xindex = np.where(x < 1)
        solutions[::2] = theta_zero[xindex]
                
        zero = rightHandSide - leftHandSideOdd
        zeros = (np.diff(np.sign(zero)) != 0)
        zeros_index = np.nonzero(zeros)
        thets = theta[zeros_index]
        leftHandSideOdd[zeros_index] = None
        
        x = -1/np.tan(thets) - np.sqrt(mW / mB * (theta0 ** 2 / thets ** 2 - 1) )
        xindex = np.where(x < 1)
        solutions[1::2] = thets[xindex]
        
        thetaf = solutions
        rightHandSide2 = np.sqrt(mW / mB * (theta0 ** 2 / thetaf ** 2 - 1) )
    
        if plot:
            fig, ax = plt.subplots(dpi = 500)
            ax.plot(theta, rightHandSide)
            ax.plot(theta, leftHandSideEven, c='y', label=r'$\tan \theta$')
            ax.plot(theta, leftHandSideOdd, c='g', label=r'$-\cot \theta$')
            plt.legend()
            ax.set_ylim(0,10)
            ax.set_xlim(0, theta[-1])
            ax.scatter(thetaf,rightHandSide2, c='r')
            ax.set_xlabel(r'$\theta$')
            plt.show()
        return thetaf




