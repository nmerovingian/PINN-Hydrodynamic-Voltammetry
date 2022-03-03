import tensorflow as tf

from .layer import GradientLayer,BoundaryGradientLayer

class PINN:


    def __init__(self, network):


        self.network = network

        self.grads = GradientLayer(self.network)
        self.boundaryGrad = BoundaryGradientLayer(self.network)



    def build(self):




        txy_dmn0 = tf.keras.layers.Input(shape=(3,))
        v_dmn0 = tf.keras.layers.Input(shape=(1,))
        txy_dmn1 = tf.keras.layers.Input(shape=(3,))
        v_dmn1 = tf.keras.layers.Input(shape=(1,))



        txy_bnd0 = tf.keras.layers.Input(shape=(3,))
        txy_bnd1 = tf.keras.layers.Input(shape=(3,))
        txy_bnd2 = tf.keras.layers.Input(shape=(3,))
        txy_bnd3 = tf.keras.layers.Input(shape=(3,))
        txy_bnd4 = tf.keras.layers.Input(shape=(3,))
        txy_bnd5 = tf.keras.layers.Input(shape=(3,))
        txy_bnd6 = tf.keras.layers.Input(shape=(3,))

        cdmn0, dc_dt_dmn0, dc_dx_dmn0,dc_dy_dmn0, d2c_dx2_dmn0,d2c_dy2_dmn0 = self.grads(txy_dmn0)
        c_dmn0 = d2c_dy2_dmn0 + d2c_dx2_dmn0 - v_dmn0*dc_dx_dmn0
        c_dmn0 = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='c_mdn0')(c_dmn0)

        cdmn1, dc_dt_dmn1, dc_dx_dmn1,dc_dy_dmn1, d2c_dx2_dmn1,d2c_dy2_dmn1 = self.grads(txy_dmn1)
        c_dmn1 = d2c_dy2_dmn1 + d2c_dx2_dmn1 - v_dmn1*dc_dx_dmn1
        c_dmn1 = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='c_mdn1')(c_dmn1)

        cbnd0,dc_dt_bnd0,dc_dx_bnd0,dc_dy_bnd0 = self.boundaryGrad(txy_bnd0)
        c_bnd0 = dc_dy_bnd0
        c_bnd0 = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='c_bnd0')(c_bnd0)

    
        c_bnd1 = self.network(txy_bnd1)
        c_bnd1 = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='c_bnd1')(c_bnd1)

        cbnd2,dc_dt_bnd2,dc_dx_bnd2,dc_dy_bnd2 = self.boundaryGrad(txy_bnd2)
        c_bnd2 = dc_dy_bnd2
        c_bnd2 = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='c_bnd2')(c_bnd2)


        c_bnd3 = self.network(txy_bnd3)
        c_bnd3 = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='c_bnd3')(c_bnd3)


        cbnd4,dc_dt_bnd4,dc_dx_bnd4,dc_dy_bnd4 = self.boundaryGrad(txy_bnd4)
        c_bnd4 = dc_dy_bnd4
        c_bnd4 = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='c_bnd4')(c_bnd4)

        c_bnd5 = self.network(txy_bnd5)
        c_bnd5 = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='c_bnd5')(c_bnd5)


        cbnd6,dc_dt_bnd6,dc_dx_bnd6,dc_dy_bnd6 = self.boundaryGrad(txy_bnd6)
        c_bnd6 = dc_dy_bnd6
        c_bnd6 = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='c_bnd6')(c_bnd6)
        















        return tf.keras.models.Model(
            inputs=[txy_dmn0,v_dmn0,txy_dmn1,v_dmn1,txy_bnd0,txy_bnd1,txy_bnd2,txy_bnd3,txy_bnd4,txy_bnd5,txy_bnd6], outputs=[c_dmn0,c_dmn1,c_bnd0,c_bnd1,c_bnd2,c_bnd3,c_bnd4,c_bnd5,c_bnd6])