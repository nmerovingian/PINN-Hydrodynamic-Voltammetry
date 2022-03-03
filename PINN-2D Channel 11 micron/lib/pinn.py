import tensorflow as tf

from .layer import GradientLayer,BoundaryGradientLayer

class PINN:


    def __init__(self, network):


        self.network = network

        self.grads = GradientLayer(self.network)
        self.boundaryGrad = BoundaryGradientLayer(self.network)



    def build(self):




        txyv_dmn0 = tf.keras.layers.Input(shape=(3,))
        v_dmn0 = tf.keras.layers.Input(shape=(1,))

        txyv_dmn1 = tf.keras.layers.Input(shape=(3,))
        v_dmn1 = tf.keras.layers.Input(shape=(1,))

        txyv_bnd0 = tf.keras.layers.Input(shape=(3,))
        txyv_bnd1 = tf.keras.layers.Input(shape=(3,))
        txyv_bnd2 = tf.keras.layers.Input(shape=(3,))
        txyv_bnd3 = tf.keras.layers.Input(shape=(3,))
        txyv_bnd4 = tf.keras.layers.Input(shape=(3,))


        cdmn0, dc_dt_dmn0, dc_dx_dmn0,dc_dy_dmn0, d2c_dx2_dmn0,d2c_dy2_dmn0 = self.grads(txyv_dmn0)
        c_dmn0 = d2c_dy2_dmn0 + d2c_dx2_dmn0 -  v_dmn0*dc_dx_dmn0
        c_dmn1 = dc_dt_dmn0

        cdmn1, dc_dt_dmn1, dc_dx_dmn1,dc_dy_dmn1, d2c_dx2_dmn1,d2c_dy2_dmn1 = self.grads(txyv_dmn1)
        c_dmn2 = d2c_dy2_dmn1 + d2c_dx2_dmn1 -  v_dmn1*dc_dx_dmn1
        c_dmn3 = dc_dt_dmn1

        c_bnd0 = self.network(txyv_bnd0)
        c_bnd1 = self.network(txyv_bnd1)


        cbnd2,dc_dt_bnd2,dc_dx_bnd2,dc_dy_bnd2 = self.boundaryGrad(txyv_bnd2)
        c_bnd2 = dc_dy_bnd2

        cbnd3,dc_dt_bnd3,dc_dx_bnd3,dc_dy_bnd3 = self.boundaryGrad(txyv_bnd3)
        c_bnd3 = dc_dy_bnd3

        cbnd4,dc_dt_bnd4,dc_dx_bnd4,dc_dy_bnd4 = self.boundaryGrad(txyv_bnd4)
        c_bnd4 = dc_dy_bnd4






        return tf.keras.models.Model(
            inputs=[txyv_dmn0,v_dmn0,txyv_dmn1,v_dmn1,txyv_bnd0,txyv_bnd1,txyv_bnd2,txyv_bnd3,txyv_bnd4], outputs=[c_dmn0,c_dmn1,c_dmn2,c_dmn3,c_bnd0,c_bnd1,c_bnd2,c_bnd3,c_bnd4])