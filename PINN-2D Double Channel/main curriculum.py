from re import X
import numpy as np 
import os
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from tensorflow.keras import callbacks
from tensorflow.python.keras.callbacks import Callback
from lib.pinn import PINN
from lib.network import Network
import pandas as pd
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(gpu,[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7680)])
from tensorflow.keras.callbacks import LearningRateScheduler
from MatsudaBraun import N_MB  

linewidth = 4
fontsize = 15

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwargs






# build a core network model
network = Network.build()
network.summary()
# build a PINN model
pinn = PINN(network).build()
pinn.compile(optimizer='Adam',loss='mse')

default_weight_name = "./weights/default.h5"
pinn.save_weights(default_weight_name)


Vfs = list()
Pes = list()

Fluxes_G = list()
Fluxes_D = list()

Es = list()
E_MBs = list()
# Dimensional Diffusion coefficient

xb_ratios  = list()
Xb_s = list()

D = 1e-9
d = 6e-3  # m 
xe= 11e-6 # m 
w = 5e-3 # m 
h = 2.5e-4 # m 
c_bulk = 1.0 # mol/cubic meter


def prediction(epochs=50,maxT=0.0,Pe=1,initial_weights = None,train=True,num_train_samples=int(1e6),plot_cv=True,saving_directory="./Data",alpha=1.0,xa=None,xb=None,xc=None):
    """
    epochs: number of epoch for the training 
    sigma: dimensionless scan rate
    train: If true, always train a new neural network. If false, use the existing weights. If weights does not exist, start training 
    plot_cv: plot the resulting CV after training 
    saving directory: where data is saved 
    """
    def schedule(epoch,lr):
        # a learning rate scheduler 
        if epoch<=400:
            return lr 
        else:
            lr *= alpha
            return lr
    # saving directory is where data(voltammogram, concentration profile etc is saved)
    if not os.path.exists(saving_directory):
        os.mkdir(saving_directory)
    
    #weights folder is where weights is saved
    if not os.path.exists('./weights'):
        os.mkdir('./weights')

    # number of training samples
    num_train_samples = num_train_samples
    # number of test samples
    num_test_samples = 1000








    H = h/xe
    Xa = xa/xe
    Xb = xb/xe
    Xc = xc/xe
    Xe = xe/xe 


    maxT = maxT
    Xsim = Xa + xe/xe + Xb+ xe/xe + Xc 
    Ysim = h*2/xe


    if Ysim/Xsim > 2.5:
        Ysim = Xsim*2.5
    


    vm = Pe*D/xe 
    vf = vm*2*h*d
    J_analytical = 0.925*(xe**2*vf/(D*h**2*d))**(1/3)


    # prefix or suffix of files 
    file_name = f'Pe={Pe:.2E} epochs={epochs:.2E} Xa = {Xa:.2E} Xb = {Xb:.2E} Xc = {Xc:.2E}n_train={num_train_samples:.2E}'


    # the training feature enoforcing fick's second law 
    # two fick's second law of diffusion scan in the X and Y direction 
    txy_dmn0 = np.random.rand(num_train_samples,3)
    txy_dmn0[:,0] = 0.0
    txy_dmn0[:,1] *= Xsim
    txy_dmn0[:,2] *= Ysim

    v_dmn0 = np.random.rand(num_train_samples,1)
    v_dmn0[:,0] = 1.5*Pe*(1.0-((H-txy_dmn0[:,2])**2.0)/(H**2.0))


    txy_dmn1 = np.random.rand(num_train_samples,3)
    txy_dmn1[:,0] = 0.0
    txy_dmn1[:,1] *= Xsim
    txy_dmn1[:,2] *= (Ysim*0.2)

    v_dmn1 = np.random.rand(num_train_samples,1)
    v_dmn1[:,0] = 1.5*Pe*(1.0-((H-txy_dmn1[:,2])**2.0)/(H**2.0))


    txy_bnd0 =  np.random.rand(num_train_samples,3)
    txy_bnd0[:,0] = 0.0
    txy_bnd0[:,1] = txy_bnd0[:,1]*Xc + (Xa+Xe+Xb+Xe)
    txy_bnd0[:,2] = 0.0

    txy_bnd1 = np.random.rand(num_train_samples,3)
    txy_bnd1[:,0] = 0.0
    txy_bnd1[:,1] = txy_bnd1[:,1]*(Xe) + (Xa+Xe+Xb)
    txy_bnd1[:,2] = 0.0


    txy_bnd2 = np.random.rand(num_train_samples,3)
    txy_bnd2[:,0] = 0.0
    txy_bnd2[:,1] = txy_bnd2[:,1]*(Xb) + (Xa+Xe)
    txy_bnd2[:,2] = 0.0


    txy_bnd3 = np.random.rand(num_train_samples,3)
    txy_bnd3[:,0] = 0.0
    txy_bnd3[:,1] = txy_bnd3[:,1]*(Xe) + Xa
    txy_bnd3[:,2] = 0.0


    txy_bnd4 = np.random.rand(num_train_samples,3)
    txy_bnd4[:,0] = 0.0
    txy_bnd4[:,1] = txy_bnd4[:,1] * Xa
    txy_bnd4[:,2] = 0.0


    txy_bnd5 = np.random.rand(num_train_samples,3)
    txy_bnd5[:,0] = 0.0 
    txy_bnd5[:,1] = 0.0
    txy_bnd5[:,2] = txy_bnd5[:,2] * Ysim


    txy_bnd6 = np.random.rand(num_train_samples,3)
    txy_bnd6[:,0] = 0.0
    txy_bnd6[:,1] = txy_bnd6[:,1]*(Xa+Xe+Xb+Xe+Xc)
    txy_bnd6[:,2] = Ysim







    c_dmn0 = np.zeros((num_train_samples,1))
    c_dmn1 = np.zeros((num_train_samples,1))
    c_bnd0 = np.zeros((num_train_samples,1))
    c_bnd1 = np.ones((num_train_samples,1))
    c_bnd2 = np.zeros((num_train_samples,1))
    c_bnd3 = np.zeros((num_train_samples,1))
    c_bnd4 = np.zeros((num_train_samples,1))
    c_bnd5 = np.ones((num_train_samples,1))
    c_bnd6 = np.zeros((num_train_samples,1))



    

    x_train = [txy_dmn0,v_dmn0,txy_dmn1,v_dmn1,txy_bnd0,txy_bnd1,txy_bnd2,txy_bnd3,txy_bnd4,txy_bnd5,txy_bnd6]
    y_train = [c_dmn0,c_dmn1,c_bnd0,c_bnd1,c_bnd2,c_bnd3,c_bnd4,c_bnd5,c_bnd6]

    


    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
    # the loss weight of each loss componentan can be varied


    lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0)
    pinn.load_weights(default_weight_name)
    # If train = True, fit the NN, else, use saved weights 
    if train:
        if initial_weights:
            pinn.load_weights(initial_weights)
        pinn.fit(x=x_train,y=y_train,epochs=epochs,batch_size =500,verbose=2,callbacks=[lr_scheduler_callback],shuffle=False)
        pinn.save_weights(f'./weights/weights {file_name}.h5')
    else:
        try:
            pinn.load_weights(f'./weights/weights {file_name}.h5')
        except:
            print('Weights does not exist\nStart training')
            if initial_weights:
                pinn.load_weights(initial_weights)
            pinn.fit(x=x_train,y=y_train,epochs=epochs,batch_size =500,verbose=2,callbacks=[lr_scheduler_callback])
            pinn.save_weights(f'./weights/weights {file_name}.h5')
    

    
    # plot concentration profile in a certain time step
    time_sects = [0.0]
    for index,time_sect in enumerate(time_sects):
        txyv_test = np.zeros((int(num_test_samples**2),3))
        txyv_test[...,0] = time_sect
        x_flat = np.linspace(0,Xsim,num_test_samples)
        y_flat = np.linspace(0,Ysim,num_test_samples)
        x,y = np.meshgrid(x_flat,y_flat)
        txyv_test[...,1] = x.flatten()
        txyv_test[...,2] = y.flatten()
        c = network.predict(txyv_test)
        c = c.reshape(x.shape)

        fig,axes = plt.subplots(figsize=(8,9),nrows=2)
        plt.subplots_adjust(hspace=0.4)
        ax = axes[0]
        mesh = ax.pcolormesh(x,y,c,shading='auto')
        cbar = plt.colorbar(mesh,pad=0.05, aspect=10,ax=ax)
        cbar.set_label('$C_A(X,Y)$',fontsize='large',fontweight='bold')
        cbar.mappable.set_clim(0, 1)
        
        ax.set_ylim(-Ysim*0.05,Ysim)
        ax.add_patch(Rectangle((0,-Ysim*0.05),Xa,Ysim*0.05,edgecolor='k',facecolor='k'))
        ax.add_patch(Rectangle((Xa,-Ysim*0.05),Xe,Ysim*0.05,edgecolor='k',facecolor='r'))
        ax.add_patch(Rectangle((Xa+Xe,-Ysim*0.05),Xa+Xe+Xb,Ysim*0.05,edgecolor='k',facecolor='k'))
        ax.add_patch(Rectangle((Xa+Xe+Xb,-Ysim*0.05),Xe,Ysim*0.05,edgecolor='k',facecolor='g'))
        ax.add_patch(Rectangle((Xa+Xe+Xb+Xe,-Ysim*0.05),Xc,Ysim*0.05,edgecolor='k',facecolor='k'))

        x_flat = np.linspace(Xa,Xa+Xe,num=500)
        x_flat_G = x_flat.copy()
        y_flat = np.linspace(0,5e-5,num=30)
        txy_flux = np.zeros((int(len(x_flat)*len(y_flat)),3))
        txy_flux[...,0] = time_sect
        x,y = np.meshgrid(x_flat,y_flat)
        txy_flux[...,1] = x.flatten()
        txy_flux[...,2] = y.flatten()

        x_i = x_flat[1] - x_flat[0]
        y_i = y_flat[1] - y_flat[0]
        c = network.predict(txy_flux)
        c = c.reshape(x.shape)
        J_G =  sum((c[20,:] - c[0,:])/ (20*y_i) * x_i)
        I_G = 96485 * w *c_bulk * D * J_G
        Flux_density_G = (c[20,:] - c[0,:])/ (20*y_i)

        x_flat = np.linspace(Xa+Xe+Xb,Xa+Xe+Xb+Xe,num=500)
        x_flat_D = x_flat.copy()
        y_flat = np.linspace(0.0,5e-5,num=30)
        txy_flux = np.zeros((int(len(x_flat)*len(y_flat)),3))
        txy_flux[...,0] = time_sect
        x,y = np.meshgrid(x_flat,y_flat)
        txy_flux[...,1] = x.flatten()
        txy_flux[...,2] = y.flatten()

        x_i = x_flat[1] - x_flat[0]
        y_i = y_flat[1] - y_flat[0]
        c = network.predict(txy_flux)
        c = c.reshape(x.shape)
        J_D =  sum((c[20,:] - c[0,:])/ (20*y_i) * x_i)
        I_D = 96485 * w *c_bulk * D * J_D
        Flux_density_D = (c[20,:] - c[0,:])/ (20*y_i)

        efficiency = -I_D/I_G
        efficiency_MB = N_MB(Xb,Xe)
        ax.set_xlabel('X-Coordinate',fontsize='large',fontweight='bold')
        ax.set_ylabel('Y-Coordinate',fontsize='large',fontweight='bold')
        ax.set_title(f'I_G = {I_G*1e6:.4f}$\mu A$ I_D = {I_D*1e6:.4f}$\mu A$\n $N_{{PINN}}$ = {-I_D/I_G:.2%} N Matsuda/Braun = {efficiency_MB:.2%}\n Pe = {Pe:.2E} $v_f$ = {vf*1e6:.6f} $cm^3/s$\n$X_b={Xb:.2f}$')


        ax = axes[1]

        ax.plot(x_flat_G,Flux_density_G,color='r',label='Generator',lw=3)
        ax.plot(x_flat_D,Flux_density_D,color='g',label='Detector',lw=3)
        ax.set_xlim(0,Xsim)
        ax.set_ylabel('Flux Density',fontsize='large',fontweight='bold')
        ax.set_xlabel('X-Coordinate',fontsize='large',fontweight='bold')
        ax.legend()





        fig.savefig(f'{saving_directory}/{file_name}.png',bbox_inches='tight')
    
    

    tf.keras.backend.clear_session()
    plt.close('all')

    Vfs.append(vf)
    Pes.append(Pe)
    Xb_s.append(Xb)
    Fluxes_G.append(J_G)
    Fluxes_D.append(J_D)
    Es.append(efficiency)
    E_MBs.append(efficiency_MB)
    

    return f'./weights/weights {file_name}.h5'







if __name__ =='__main__':
    
    maxT = 0.0
    Pe = 100

    epochs = 100
    saving_directory = 'compare with analytical'


    for xb in [1e-6,2e-6,3e-6,4e-6,5e-6,6e-6,7e-6,8e-6,9e-6,10e-6,11e-6]:
        weights = prediction(epochs = epochs,maxT=0.0,Pe=Pe,initial_weights=None,train=False,saving_directory=saving_directory,xa=2e-6,xb=xb,xc=2e-6)
    



    df = pd.DataFrame({'Pe':Pes,'Vf':Vfs,'Xb_s':Xb_s,'Flux_G':Fluxes_G,"Flux_D":Fluxes_D,"Efficiency":Es,'Efficiency MB':E_MBs})
    df.to_csv(f'Summary {saving_directory}.csv',index=False)

    
    ## plot the analytical expression with PINN prediction
    df = pd.read_csv(f'Summary {saving_directory}.csv')

    fig,ax = plt.subplots(figsize=(8,4.5))
    ax.plot(df['Xb_s'],df['Efficiency'],label='PINN',marker='o',lw=3,color='b',markersize=6,alpha=0.7)
    ax.plot(df['Xb_s'],df['Efficiency MB'],label='Matsuda/Braun Expression',lw=3,color='r',ls='--',alpha=0.7)


    ax.legend()
    ax.set_xlabel('$X_b$',fontsize='large',fontweight='bold')
    ax.set_ylabel('$N$',fontsize='large',fontweight='bold')
    ax.set_ylim([0.1,0.5])

    fig.savefig('compare with analytical.png',bbox_inches='tight')





    
    
    



  