import macroC as mC
import numpy as np
import os
import sys
from filelock import FileLock

#betaslist=list(np.linspace(10,100,50))
#betaslist = list(np.linspace(0.001,1,25))+list(np.linspace(1,10,25))
#betaslist=list(np.linspace(0.001,10,50))
betaslist=list(np.linspace(10,20,50))
nbetas=len(betaslist)
k=int(sys.argv[1])
q = k//nbetas
r = k%nbetas
betas=[betaslist[r]]
shots=int(sys.argv[4])
spins=int(sys.argv[2])
J=int(sys.argv[3])
H=mC.systemCN(spins,J)[0]
layers=int(sys.argv[6])
method=str(sys.argv[5])

truncn=q

types=['NoTrunc/','Trunc1/','Trunc2/','Trunc3/','Trunc4/']
typetrunc=types[truncn]

if shots==0:
    shots=None

if truncn==0:
    results_expect, energies_expect, thetaExpect, niterExpect, tExpect=mC.minimize_function(H,mC.expectation_classic,betas,shots=shots, layers=layers, method=method)
    results_free_energy, energies_free_energy, theta_free_energy, niter_free_energy, t_free_energy=mC.minimize_function(H,mC.free_energy,betas,shots=shots, layers=layers, method=method)
elif truncn==1:
    results_expect, energies_expect, thetaExpect, niterExpect, tExpect=mC.minimize_function_trunc(H,mC.expectation_classic,betas,shots=shots, layers=layers, method=method)
    results_free_energy, energies_free_energy, theta_free_energy, niter_free_energy, t_free_energy=mC.minimize_function_trunc(H,mC.free_energy,betas,shots=shots, layers=layers, method=method)
elif truncn==2:
    results_expect, energies_expect, thetaExpect, niterExpect, tExpect=mC.minimize_function_trunc2(H,mC.expectation_classic,betas,shots=shots, layers=layers, method=method)
    results_free_energy, energies_free_energy, theta_free_energy, niter_free_energy, t_free_energy=mC.minimize_function_trunc2(H,mC.free_energy,betas,shots=shots, layers=layers, method=method)
elif truncn==3:
    results_expect, energies_expect, thetaExpect, niterExpect, tExpect=mC.minimize_function_trunc3(H,mC.expectation_classic,betas,shots=shots, layers=layers, method=method)
    results_free_energy, energies_free_energy, theta_free_energy, niter_free_energy, t_free_energy=mC.minimize_function_trunc3(H,mC.free_energy,betas,shots=shots, layers=layers, method=method)
elif truncn==4:
    results_expect, energies_expect, thetaExpect, niterExpect, tExpect=mC.minimize_function_trunc4(H,mC.expectation_classic,betas,shots=shots, layers=layers, method=method)
    results_free_energy, energies_free_energy, theta_free_energy, niter_free_energy, t_free_energy=mC.minimize_function_trunc4(H,mC.free_energy,betas,shots=shots, layers=layers, method=method)

directoryFE = '../Data/'+'FE/'+typetrunc
directoryH = '../Data/'+'H/'+typetrunc

name=str(spins)+'SpinsCN-J='+str(J)
pathFE=directoryFE+name+'-'+method+'-'+'Layers='+str(layers)+'-'+'Shots='+str(shots)
pathH=directoryH+name+'-'+method+'-'+'Layers='+str(layers)+'-'+'Shots='+str(shots)
fileFE=pathFE+'.dat'
fileH=pathH+'.dat'

dataFE=[betas[0],results_free_energy[0],niter_free_energy[0],t_free_energy[0]]
dataFE=dataFE+list(energies_free_energy[0])
dataFE.append(len(theta_free_energy[0]))
dataFE=dataFE+list(theta_free_energy[0])


dataH=[betas[0],results_expect[0],niterExpect[0],tExpect[0]]
dataH=dataH+list(energies_expect[0])
dataH.append(len(thetaExpect[0]))
dataH=dataH+list(thetaExpect[0])

with FileLock(pathFE+'.lock'):
    if not os.path.exists(fileFE):
        with open(fileFE,'w') as file:
            np.savetxt(file,[dataFE])
            file.flush()
    else:
        with open(fileFE,'a') as file:
            np.savetxt(file,[dataFE])
            file.flush()
 

with FileLock(pathH+'.lock'):
    if not os.path.exists(fileH):
        with open(fileH,'w') as file:
            np.savetxt(file,[dataH])
            file.flush()
    else:
        with open(fileH,'a') as file:
            np.savetxt(file,[dataH])
            file.flush()
   