import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import sys


from parameters import params

class QuantumWell():
    C = 1.602176565e-19 #C #Charge of electron
    hb = 1.054571817e-34 #Js #Reduced Planck's constant
    m0 = 9.10938291e-31 #kg #Rest mass of electron
    def __init__(self, a, well='GaAs', barrier='AlAs', phase='cub', plot=True):

        self.setup_parameters(a, well, barrier, phase, plot)

        
        #self.normalize(self.bandEc, self.Ec, self.kc, self.thetac)
        
    def setup_parameters(self, aW, well, barrier, phase, plot):
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
        self.aW = self.check_aW(aW)
        self.setup(self.aW, self.plot)
        
    def check_aW(self, aW):

        '''
        Converts aW to numpy array in meters
        
        aW: int or float in nm
        '''

        try:
            if aW.shape[0] > 1:
                self.plot= False
        except:
            try:
                aW = np.arange(aW, aW + 0.99, 1.0)
                
            except:
                print('Wrong format for aW')
                sys.exit()
        aW = np.round(aW, 4)

        return aW
        
    def setup(self, aW, plot): 
        for a in aW:
            #Prepare containers for energies, k, and theta for each aW, for each band.
            self.E[str(a)] = {}
            self.E[str(a)]['c'] = []
            self.E[str(a)]['v'] = []
            self.k[str(a)] = {}
            self.k[str(a)]['c'] = []
            self.k[str(a)]['v'] = []
            self.t[str(a)] = {}
            self.t[str(a)]['c'] = []
            self.t[str(a)]['v'] = []
            
            Ec, kc, tc = self.energy_calculator(self.V0['c'] * self.C, a * 1e-9, 'c' , plot)
            Ev, kv, tv = self.energy_calculator(self.V0['v'] * self.C, a * 1e-9, 'v', plot)
            
            self.E[str(a)]['c'] = Ec
            self.E[str(a)]['v'] = Ev 
            self.k[str(a)]['c'] = kc
            self.k[str(a)]['v'] = kv
            self.t[str(a)]['c'] = tc
            self.t[str(a)]['v'] = tv

            self.dE.append(Ec[0] + self.EgW + Ev[0])

        
        #if self.aW.shape[0] > 1:
             #fig, ax = plt.subplots(dpi=600)
             #plt.plot(self.aW, dE)
    def plot_aW(self):
        if self.aW.shape[0] > 1:
            fig, ax = plt.subplots(dpi=500)
            ax.plot(self.aW, self.dE)
            ax.set_xlabel(r'$L_W$')
            ax.set_ylabel(r'Emission')
       
    def energy_calculator(self, V0, aW, band, plot):
        C = 1.602176565e-19 #C #Charge of electron
        hb = 1.054571817e-34 #Js #Reduced Planck's constant
        m0 = 9.10938291e-31 #kg #Rest mass of electron

        mW = params[self.well][self.phase][self.m[band]]
        mB = params[self.barrier][self.phase][self.m[band]]
        theta0 = np.sqrt(m0 * mW * V0 * aW ** 2 / (2 * hb ** 2) )
        NoOS = np.ceil(theta0/(np.pi/2));
        
        theta = np.arange(self.dtheta, theta0, self.dtheta);
        
        rightHandSide = np.sqrt( mW / mB * (theta0 ** 2 / theta ** 2 - 1) )
        leftHandSideEven = np.tan(theta)
        leftHandSideOdd = -1/np.tan(theta)

        theta = self.intersection_finder(NoOS, theta, mW, mB, theta0, rightHandSide, 
                                     leftHandSideEven, leftHandSideOdd, plot)
    
        theta = np.sort(theta)
        k = 2 * theta / aW
        E = hb ** 2 * k ** 2 / (2*m0*mW*C)
        return E, k, theta
    
    def normalize(self, band, a=None, plot=False):
        '''
        band: str
        a: int or float in nm
        '''
        if band !='c' and band!='v':
            print('Incorrect band \'c\' or \'v\'')
            return
        if a == None:
            aW = self.aW[0]
        else:
            if str(float(a)) not in self.E.keys():
                self.setup(self.check_aW(a), plot)
            aW = a
        V0 = self.V0[band]
        mW = params[self.well][self.phase][self.m[band]]
        mB = params[self.barrier][self.phase][self.m[band]]
        
        k = self.k[str(float(aW))][band]
        theta = self.t[str(float(aW))][band]
        E = self.E[str(float(aW))][band]

        A = np.zeros(k.shape[0])
        B = np.zeros(k.shape[0])

        k=k*1e-9 #To nm^-1
        
        V0 = V0
        kappa = np.zeros(k.shape[0])
        kappa[::2] = k[::2]*mB/mW*np.tan(theta[::2])
        kappa[1::2] = -k[1::2]*mB/mW/np.tan(theta[1::2])
        #Ec 2.111 Quantum wells, wires and dots Harrison
        A[::2] = 1/np.sqrt(aW/2 + np.sin(k[::2]*aW)/(2 * k[::2]) + np.cos(k[::2]*aW/2) ** 2 / kappa[::2]) 
        B[::2] = A[::2] * np.exp(kappa[::2] * aW / 2) * np.cos(k[::2] * aW / 2)
        #Ec 2.112 Quantum wells, wires and dots Harrison
        A[1::2] = 1/np.sqrt(aW/2 - np.sin(k[1::2]*aW)/(2 * k[1::2]) + np.sin(k[1::2]*aW/2) ** 2 / kappa[1::2]) 
        B[1::2] = A[1::2] * np.exp(kappa[1::2] * aW / 2) * np.sin(k[1::2] * aW / 2)
        aWs = aW/1000 #aW step
        aWl = []
        aWrange = 3*aW
        for _ in k:
            aWl.append(np.arange(-aWrange, aWrange + aWs, aWs))
        aWr = np.array(aWl)
        aWn = int((aWrange - aW/2)/aWs) #index of -aW
        aWp = int((aWrange + aW/2)/aWs)
        
        psi = np.zeros((k.shape[0],aWr.shape[1]))
        psi[::2,:aWn] = (np.exp(aWr[::2].T[:aWn,:]*(kappa[::2]))*B[::2]).T
        psi[::2,aWn:aWp] = (np.cos(aWr[::2].T[aWn:aWp,:]*(k[::2]))*A[::2]).T
        psi[::2,aWp:] = (np.exp(-aWr[::2].T[aWp:,:]*(kappa[::2]))*B[::2]).T
        
        psi[1::2,:aWn] = (np.exp(aWr[1::2].T[:aWn,:]*(kappa[1::2]))*(-B[1::2])).T
        psi[1::2,aWn:aWp] = (np.sin(aWr[1::2].T[aWn:aWp,:]*(k[1::2]))*A[1::2]).T
        psi[1::2,aWp:] = (np.exp(-aWr[1::2].T[aWp:,:]*(kappa[1::2]))*B[1::2]).T
        
        normalization = (integrate.simpson(psi ** 2, dx=aWs)) # psi - nergies for the integration and confirm normalization

        print(normalization)
        
        fig, ax = plt.subplots(dpi=500)
        ax.plot(aWr.T, psi.T + E, linewidth=0.5)
        wellx = [-aWrange, -aW/2, -aW/2, aW/2, aW/2, aWrange]
        welly = [V0, V0, 0,0, V0, V0]
        ax.set_ylabel(r'$E$')
        ax.set_xlabel(r'$L$')
        plt.plot(wellx, welly, color='black', linewidth=1)
        energyx = [-aWrange, aWrange]
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




