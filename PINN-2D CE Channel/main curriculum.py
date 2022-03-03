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
"""
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(gpu,[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7680)])
"""
from tensorflow.keras.callbacks import LearningRateScheduler

linewidth = 4
fontsize = 15

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwargs






# build a core network model
networkR = Network.build()
networkA = Network.build()
networkR.summary()
# build a PINN model
pinn = PINN(networkR,networkA).build()
pinn.compile(optimizer='Adam',loss='mse')

default_weight_name = "./weights/default.h5"
pinn.save_weights(default_weight_name)


Vfs = list()
Pes = list()
a_s = list()
b_s = list()
Kfs = list()
Kbs = list()
Keqs = list()
n_train_s = list()
Fluxes = list()
Fluxes_Levich  = list()

# Dimensional Diffusion coefficient
D = 2.3e-9
d = 6e-3  # m 
xe= 11e-6 # m 
w = 4.7e-3 # m 
h = 2.5e-4 # m 
c_bulk = 1.14 # mol/cubic meter


def prediction(epochs=50,maxT=0.0,Pe=1,Kf=1.0,Kb=1.0,Keq=1.0,num_train_samples=1e6,initial_weights = None,train=True,plot_cv=True,saving_directory="./Data",alpha=1.0,a=0.1,b=0.1):
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
    num_train_samples = int(num_train_samples)
    # number of test samples
    num_test_samples = 1000









    maxT = maxT
    Xsim = xe/xe
    Ysim = h*2/xe
    before_electrode = a
    after_electrode = b


    CR_bulk = 1.0 /(1.0+Keq)
    CA_bulk = Keq/(1.0+Keq)





    # prefix or suffix of files 
    file_name = f'Pe={Pe:.2E} maxT={maxT:.2E} epochs={epochs:.2E} a = {before_electrode:.2E} b = {after_electrode:.2E} n_train={num_train_samples:.2E} Kf={Kf:.2E} Kb={Kb:.2E} Keq={Keq:.2E}'

    if Ysim/Xsim > 10:
        Ysim = Xsim*10
    
    H = h/xe

    vm = Pe*D/xe 
    vf = vm*2*h*d
    J_Levich = 0.925*(xe**2*vf/(D*h**2*d))**(1/3)




    # the training feature enoforcing fick's second law 
    # two fick's second law of diffusion scan in the X and Y direction 
    txyv_dmn0 = np.random.rand(num_train_samples,3)
    txyv_dmn0[:,0] = 0.0
    txyv_dmn0[:,1] = txyv_dmn0[:,1]*(Xsim + before_electrode+after_electrode)
    txyv_dmn0[:,2] *= Ysim

    v_dmn0 = np.random.rand(num_train_samples,1)
    v_dmn0[:,0] = 1.5*Pe*(1.0-((H-txyv_dmn0[:,2])**2.0)/(H**2.0))

    Kf_dmn0 = np.random.rand(num_train_samples,1)
    Kf_dmn0[:,0] = Kf 
    Kb_dmn0 = np.random.rand(num_train_samples,1)
    Kb_dmn0[:,0] = Kb





    txyv_bnd0 = np.random.rand(num_train_samples,3)
    txyv_bnd0[:,0] = 0.0
    txyv_bnd0[:,1] = Xsim *txyv_bnd0[:,1] + before_electrode
    txyv_bnd0[:,2] = 0.0
    #txyv_bnd0[:,3] = 1.5*Pe*(1.0-((H-txyv_bnd0[:,2])**2)/(H**2))


    txyv_bnd1 = np.random.rand(num_train_samples,3)
    txyv_bnd1[:,0] = 0.0
    txyv_bnd1[:,1] = 0.0
    txyv_bnd1[:,2] *= Ysim
    #txyv_bnd1[:,3] = 1.5*Pe*(1.0-((H-txyv_bnd1[:,2])**2)/(H**2))

    txyv_bnd2 = np.random.rand(num_train_samples,3)
    txyv_bnd2[:,0] = 0.0
    txyv_bnd2[:,1] *= (Xsim+before_electrode + after_electrode)
    txyv_bnd2[:,2] = Ysim
    #txyv_bnd2[:,3] = 1.5*Pe*(1.0-((H-txyv_bnd2[:,2])**2)/(H**2))

    txyv_bnd3 = np.random.rand(num_train_samples,3)
    txyv_bnd3[:,0] = 0.0
    txyv_bnd3[:,1] *= before_electrode
    txyv_bnd3[:,2] = 0.0


    txyv_bnd4 = np.random.rand(num_train_samples,3)
    txyv_bnd4[:,0] = 0.0
    txyv_bnd4[:,1] = txyv_bnd4[:,1]*after_electrode + (Xsim+before_electrode)
    txyv_bnd4[:,2] = 0.0



    c_dmn0R = np.zeros((num_train_samples,1))
    c_dmn0A = np.zeros((num_train_samples,1))


    c_dmn1R = np.zeros((num_train_samples,1))
    c_dmn1A = np.zeros((num_train_samples,1))

    c_bnd0R = np.zeros((num_train_samples,1))
    c_bnd0A = np.zeros((num_train_samples,1))

    c_bnd1R = np.ones((num_train_samples,1)) # fixed concentration at the left of the channel
    c_bnd1A = np.ones((num_train_samples,1))
    c_bnd1R[:,0] = CR_bulk
    c_bnd1A[:,0] = CA_bulk



    c_bnd2R = np.zeros((num_train_samples,1))
    c_bnd2A = np.zeros((num_train_samples,1))
    c_bnd3R = np.zeros((num_train_samples,1))
    c_bnd3A = np.zeros((num_train_samples,1))
    c_bnd4R = np.zeros((num_train_samples,1))
    c_bnd4A = np.zeros((num_train_samples,1))
    #c_ini = np.ones((num_train_samples,1))

    

    x_train = [txyv_dmn0,v_dmn0,Kf_dmn0,Kb_dmn0,txyv_bnd0,txyv_bnd1,txyv_bnd2,txyv_bnd3,txyv_bnd4]
    y_train = [c_dmn0R,c_dmn0A,c_bnd0R,c_bnd0A,c_bnd1R,c_bnd1A,c_bnd2R,c_bnd2A,c_bnd3R,c_bnd3A,c_bnd4R,c_bnd4A]

    


    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
    # the loss weight of each loss componentan can be varied


    lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0)
    pinn.load_weights(default_weight_name)
    # If train = True, fit the NN, else, use saved weights 
    if train:
        if initial_weights:
            pinn.load_weights(initial_weights)
        pinn.fit(x=x_train,y=y_train,epochs=epochs,batch_size =1000,verbose=2,callbacks=[lr_scheduler_callback],shuffle=False)
        pinn.save_weights(f'./weights/weights {file_name}.h5')
    else:
        try:
            pinn.load_weights(f'./weights/weights {file_name}.h5')
        except:
            print('Weights does not exist\nStart training')
            if initial_weights:
                pinn.load_weights(initial_weights)
            pinn.fit(x=x_train,y=y_train,epochs=epochs,batch_size =1000,verbose=2,callbacks=[lr_scheduler_callback])
            pinn.save_weights(f'./weights/weights {file_name}.h5')
    

    
    # plot concentration profile in a certain time step
    time_sects = [0.0]
    for index,time_sect in enumerate(time_sects):
        txyv_test = np.zeros((int(num_test_samples**2),3))
        txyv_test[...,0] = time_sect
        x_flat = np.linspace(0,Xsim+before_electrode+after_electrode,num_test_samples)
        y_flat = np.linspace(0,Ysim,num_test_samples)
        x,y = np.meshgrid(x_flat,y_flat)
        txyv_test[...,1] = x.flatten()
        txyv_test[...,2] = y.flatten()
        c = networkR.predict(txyv_test)
        c = c.reshape(x.shape)
        print('CR avergae',np.average(c))

        fig,axes = plt.subplots(figsize=(8,14),nrows=3)
        plt.subplots_adjust(hspace=0.6)
        ax = axes[0]
        mesh = ax.pcolormesh(x,y,c,shading='auto')
        cbar = plt.colorbar(mesh,pad=0.05, aspect=10,ax=ax)
        cbar.set_label('$C_R(X,Y)$')
        cbar.mappable.set_clim(0, CR_bulk)
        
        ax.set_ylim(-Ysim*0.05,Ysim)
        ax.add_patch(Rectangle((before_electrode,-Ysim*0.05),Xsim,Ysim*0.05,edgecolor='k',facecolor='r'))
        ax.add_patch(Rectangle((0.0,-Ysim*0.05),before_electrode,Ysim*0.05,edgecolor='k',facecolor='k'))
        ax.add_patch(Rectangle((Xsim+before_electrode,-Ysim*0.05),after_electrode,Ysim*0.05,edgecolor='k',facecolor='k'))
        x_flat = np.linspace(0.0,Xsim+before_electrode+after_electrode,num=500)
        y_flat = np.linspace(0,5e-5,num=30)
        txy_flux = np.zeros((int(len(x_flat)*len(y_flat)),3))
        txy_flux[...,0] = time_sect
        x,y = np.meshgrid(x_flat,y_flat)
        txy_flux[...,1] = x.flatten()
        txy_flux[...,2] = y.flatten()

        x_i = x_flat[1] - x_flat[0]
        y_i = y_flat[1] - y_flat[0]
        c = networkR.predict(txy_flux)
        c = c.reshape(x.shape)
        J =  sum((c[20,:] - c[0,:])/ (20*y_i) * x_i)
        I_PINN = 96485 * w *c_bulk * D * J

        ax.set_title(f'Flux = {J:.2f} Current = {I_PINN*1e6:.4f} $\mu A$\n Pe = {Pe:.2E} $v_f$ = {vf*1e6:.4f} $cm^3/s$\n$K_f={Kf:.2E}$,$K_B={Kb:.2E}$\n')


        ax = axes[1]
        txyv_test = np.zeros((int(num_test_samples**2),3))
        txyv_test[...,0] = time_sect
        x_flat = np.linspace(0,Xsim+before_electrode+after_electrode,num_test_samples)
        y_flat = np.linspace(0,Ysim,num_test_samples)
        x,y = np.meshgrid(x_flat,y_flat)
        txyv_test[...,1] = x.flatten()
        txyv_test[...,2] = y.flatten()
        c = networkA.predict(txyv_test)
        print('CA avergae',np.average(c))

        c = c.reshape(x.shape)
        mesh = ax.pcolormesh(x,y,c,shading='auto')
        cbar = plt.colorbar(mesh,pad=0.05, aspect=10,ax=ax)
        cbar.set_label('$C_A(X,Y)$')
        cbar.mappable.set_clim(0, CA_bulk)
        
        ax.set_ylim(-Ysim*0.05,Ysim)
        ax.add_patch(Rectangle((before_electrode,-Ysim*0.05),Xsim,Ysim*0.05,edgecolor='k',facecolor='r'))
        ax.add_patch(Rectangle((0.0,-Ysim*0.05),before_electrode,Ysim*0.05,edgecolor='k',facecolor='k'))
        ax.add_patch(Rectangle((Xsim+before_electrode,-Ysim*0.05),after_electrode,Ysim*0.05,edgecolor='k',facecolor='k'))
        x_flat = np.linspace(0.0,Xsim+before_electrode+after_electrode,num=500)
        y_flat = np.linspace(0,5e-5,num=30)
        txy_flux = np.zeros((int(len(x_flat)*len(y_flat)),3))
        txy_flux[...,0] = time_sect
        x,y = np.meshgrid(x_flat,y_flat)
        txy_flux[...,1] = x.flatten()
        txy_flux[...,2] = y.flatten()

        x_i = x_flat[1] - x_flat[0]
        y_i = y_flat[1] - y_flat[0]
        c = networkA.predict(txy_flux)

        c = c.reshape(x.shape)
        J =  sum((c[20,:] - c[0,:])/ (20*y_i) * x_i)
        I_PINN = 96485 * w *c_bulk * D * J

        ax.set_title(f'Flux = {J:.2f} Current = {I_PINN*1e6:.4f} $\mu A$\n Pe = {Pe:.2E} $v_f$ = {vf*1e6:.4f} $cm^3/s$\n$K_f={Kf:.2E}$,$K_B={Kb:.2E}$')

        ax = axes[2]
        Flux_density = (c[20,:] - c[0,:])/ (20*y_i)
        ax.plot(x_flat,Flux_density)
        ax.set_xlim(0,Xsim+before_electrode+after_electrode)
        ax.set_ylabel('Flux Density',fontsize='large')
        ax.set_xlabel('X-axis',fontsize='large')




        fig.savefig(f'{saving_directory}/Pe={Pe:.2f} epochs ={epochs} Kf={Kf:.2E} Kb={Kb:.2E} a={before_electrode:.2E} b = {after_electrode:.2E} n_train={num_train_samples}.png')
    
    


    tf.keras.backend.clear_session()
    plt.close('all')

    Vfs.append(vf)
    Pes.append(Pe)
    Fluxes.append(J)
    Fluxes_Levich.append(J_Levich)
    a_s.append(before_electrode)
    b_s.append(after_electrode)
    Kfs.append(Kf)
    Kbs.append(Kb)
    Keqs.append(Keq)
    n_train_s.append(num_train_samples)
    return f'./weights/weights {file_name}.h5'





if __name__ =='__main__':
    
    maxT = 0.0





    Pe = 15.942
    epochs = 400


    
    


    

    
    
    epochs = 400
    saving_directory = "Keq=0.5 Pe=6.38"
    Keq = 0.5
    weights = prediction(epochs=epochs,maxT=maxT,Pe=Pe*0.4,Kf=0.0,Kb=0.0,Keq=Keq,num_train_samples=2e6,train=False,initial_weights = None,saving_directory=saving_directory,a=0.2,b=0.2)
    weights = prediction(epochs=epochs,maxT=maxT,Pe=Pe*0.4,Kf=1.0,Kb=1.0/Keq,Keq=Keq,num_train_samples=2e6,train=False,initial_weights = None,saving_directory=saving_directory,a=0.2,b=0.2)
    for Kf in [2,3,4,5,6,7,8,9,10,15,20,25,30,40,50,60,70,80,90,100]:
        weights = prediction(epochs=epochs,maxT=maxT,Pe=Pe*0.4,Kf=Kf,Kb=Kf/Keq,Keq=Keq,num_train_samples=2e6,train=False,initial_weights = None,saving_directory=saving_directory,a=0.2,b=0.2) #good
        df = pd.DataFrame({'Pe':Pes,'Vf':Vfs,'a':a_s,'b':b_s,'Kf':Kfs,'Kb':Kbs,'Keq':Keqs,'n_train':n_train_s,'Flux':Fluxes,'Flux Levich':Fluxes_Levich})
        df.to_csv(f'Summary {saving_directory}.csv',index=False)
    

    Vfs = list()
    Pes = list()
    a_s = list()
    b_s = list()
    Kfs = list()
    Kbs = list()
    Keqs = list()
    n_train_s = list()
    Fluxes = list()
    Fluxes_Levich = list()
    
    
    
    

    
    

    


    
    
    epochs = 200
    saving_directory = "Keq=0.5 Pe=1.59"
    Keq = 0.5
    weights = prediction(epochs=epochs,maxT=maxT,Pe=Pe*0.1,Kf=0.0,Kb=0.0,Keq=Keq,num_train_samples=2e6,train=False,initial_weights = None,saving_directory=saving_directory,a=0.3,b=0.3)
    weights = prediction(epochs=epochs,maxT=maxT,Pe=Pe*0.1,Kf=1.0,Kb=1.0/Keq,Keq=Keq,num_train_samples=2e6,train=False,initial_weights = None,saving_directory=saving_directory,a=0.3,b=0.3)
    for Kf in [2,3,4,5,6,7,8,9,10,15,20,25,30,]:
        weights = prediction(epochs=epochs,maxT=maxT,Pe=Pe*0.1,Kf=Kf,Kb=Kf/Keq,Keq=Keq,num_train_samples=2e6,train=False,initial_weights = None,saving_directory=saving_directory,a=0.3,b=0.3) #good

    epochs = 200
    for Kf in [40,50,60,70,80,90]: 
        weights = prediction(epochs=epochs,maxT=maxT,Pe=Pe*0.1,Kf=Kf,Kb=Kf/Keq,Keq=Keq,num_train_samples=2e6,train=False,initial_weights = None,saving_directory=saving_directory,a=0.6,b=0.6)
        df = pd.DataFrame({'Pe':Pes,'Vf':Vfs,'a':a_s,'b':b_s,'Kf':Kfs,'Kb':Kbs,'Keq':Keqs,'n_train':n_train_s,'Flux':Fluxes,'Flux Levich':Fluxes_Levich})
        df.to_csv(f'Summary {saving_directory}.csv',index=False)
    for Kf in [100]: 
        weights = prediction(epochs=epochs,maxT=maxT,Pe=Pe*0.1,Kf=Kf,Kb=Kf/Keq,Keq=Keq,num_train_samples=2e6,train=False,initial_weights = None,saving_directory=saving_directory,a=0.7,b=0.7)
        df = pd.DataFrame({'Pe':Pes,'Vf':Vfs,'a':a_s,'b':b_s,'Kf':Kfs,'Kb':Kbs,'Keq':Keqs,'n_train':n_train_s,'Flux':Fluxes,'Flux Levich':Fluxes_Levich})
        df.to_csv(f'Summary {saving_directory}.csv',index=False)

    Vfs = list()
    Pes = list()
    a_s = list()
    b_s = list()
    Kfs = list()
    Kbs = list()
    Keqs = list()
    n_train_s = list()
    Fluxes = list()
    Fluxes_Levich = list()
    
    
