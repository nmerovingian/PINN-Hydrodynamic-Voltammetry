import tensorflow as tf

from .layer import GradientLayer,BoundaryGradientLayer

class PINN:


    def __init__(self, networkR,networkA):


        self.networkR = networkR
        self.networkA = networkA

        self.gradsR = GradientLayer(self.networkR)
        self.boundaryGradR = BoundaryGradientLayer(self.networkR)
        self.gradsA = GradientLayer(self.networkA)
        self.boundaryGradA = BoundaryGradientLayer(self.networkA)






    def build(self):





        txyv_dmn0 = tf.keras.layers.Input(shape=(3,))
        v_dmn0 = tf.keras.layers.Input(shape=(1,))
        Kf_dmn0 = tf.keras.layers.Input(shape=(1,))
        Kb_dmn0 = tf.keras.layers.Input(shape=(1,))

        #txyv_dmn1 = tf.keras.layers.Input(shape=(3,))
        #v_dmn1 = tf.keras.layers.Input(shape=(1,))

        txyv_bnd0 = tf.keras.layers.Input(shape=(3,))
        txyv_bnd1 = tf.keras.layers.Input(shape=(3,))
        txyv_bnd2 = tf.keras.layers.Input(shape=(3,))
        txyv_bnd3 = tf.keras.layers.Input(shape=(3,))
        txyv_bnd4 = tf.keras.layers.Input(shape=(3,))

        
        cdmn0R, dc_dt_dmn0R, dc_dx_dmn0R,dc_dy_dmn0R, d2c_dx2_dmn0R,d2c_dy2_dmn0R = self.gradsR(txyv_dmn0)
        cdmn0A, dc_dt_dmn0A, dc_dx_dmn0A,dc_dy_dmn0A, d2c_dx2_dmn0A,d2c_dy2_dmn0A = self.gradsA(txyv_dmn0)
        c_dmn0R = d2c_dy2_dmn0R + d2c_dx2_dmn0R -  v_dmn0*dc_dx_dmn0R - Kf_dmn0*cdmn0R + Kb_dmn0*cdmn0A
        c_dmn0A = d2c_dy2_dmn0A + d2c_dx2_dmn0A -  v_dmn0*dc_dx_dmn0A + Kf_dmn0*cdmn0R - Kb_dmn0*cdmn0A
        c_dmn0R = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='c_dmn0R')(c_dmn0R)
        c_dmn0A = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='c_dmn0A')(c_dmn0A)

        """
        cdmn1R, dc_dt_dmn1R, dc_dx_dmn1R,dc_dy_dmn1R, d2c_dx2_dmn1R,d2c_dy2_dmn1R = self.gradsR(txyv_dmn1)
        cdmn1A, dc_dt_dmn1A, dc_dx_dmn1A,dc_dy_dmn1A, d2c_dx2_dmn1A,d2c_dy2_dmn1A = self.gradsA(txyv_dmn1)
        c_dmn1R = d2c_dy2_dmn1R + d2c_dx2_dmn1R -  v_dmn1*dc_dx_dmn1R - self.Kf*cdmn1R + self.Kb*cdmn1A
        c_dmn1A = d2c_dy2_dmn1A + d2c_dx2_dmn1A -  v_dmn1*dc_dx_dmn1A + self.Kf*cdmn1R - self.Kb*cdmn1A
        c_dmn1R = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='c_dmn1R')(c_dmn1R)
        c_dmn1A = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='c_dmn1A')(c_dmn1A)

        """

        cbnd0R,dc_dt_bnd0R,dc_dx_bnd0R,dc_dy_bnd0R = self.boundaryGradR(txyv_bnd0)
        cbnd0A,dc_dt_bnd0A,dc_dx_bnd0A,dc_dy_bnd0A = self.boundaryGradA(txyv_bnd0)
        c_bnd0R = dc_dy_bnd0R
        c_bnd0A = cbnd0A
        c_bnd0R = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='c_bnd0R')(c_bnd0R)
        c_bnd0A = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='c_bnd0A')(c_bnd0A)

        c_bnd1R = self.networkR(txyv_bnd1)
        c_bnd1A = self.networkA(txyv_bnd1)
        c_bnd1R = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='c_bnd1R')(c_bnd1R)
        c_bnd1A = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='c_bnd1A')(c_bnd1A)



        cbnd2R,dc_dt_bnd2R,dc_dx_bnd2R,dc_dy_bnd2R = self.boundaryGradR(txyv_bnd2)
        cbnd2A,dc_dt_bnd2A,dc_dx_bnd2A,dc_dy_bnd2A = self.boundaryGradA(txyv_bnd2)
        c_bnd2R = dc_dy_bnd2R
        c_bnd2A = dc_dy_bnd2A
        c_bnd2R = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='c_bnd2R')(c_bnd2R)
        c_bnd2A = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='c_bnd2A')(c_bnd2A)



        cbnd3R,dc_dt_bnd3R,dc_dx_bnd3R,dc_dy_bnd3R = self.boundaryGradR(txyv_bnd3)
        cbnd3A,dc_dt_bnd3A,dc_dx_bnd3A,dc_dy_bnd3A = self.boundaryGradA(txyv_bnd3)
        c_bnd3R = dc_dy_bnd3R
        c_bnd3A = dc_dy_bnd3A
        c_bnd3R = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='c_bnd3R')(c_bnd3R)
        c_bnd3A = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='c_bnd3A')(c_bnd3A)



        cbnd4R,dc_dt_bnd4R,dc_dx_bnd4R,dc_dy_bnd4R = self.boundaryGradR(txyv_bnd4)
        cbnd4A,dc_dt_bnd4A,dc_dx_bnd4A,dc_dy_bnd4A = self.boundaryGradA(txyv_bnd4)
        c_bnd4R = dc_dy_bnd4R
        c_bnd4A = dc_dy_bnd4A
        c_bnd4R = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='c_bnd4R')(c_bnd4R)
        c_bnd4A = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='c_bnd4A')(c_bnd4A)




        return tf.keras.models.Model(
            inputs=[txyv_dmn0,v_dmn0,Kf_dmn0,Kb_dmn0,txyv_bnd0,txyv_bnd1,txyv_bnd2,txyv_bnd3,txyv_bnd4], outputs=[c_dmn0R,c_dmn0A,c_bnd0R,c_bnd0A,c_bnd1R,c_bnd1A,c_bnd2R,c_bnd2A,c_bnd3R,c_bnd3A,c_bnd4R,c_bnd4A])