import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, concatenate, Lambda
from tensorflow.keras.utils import to_categorical
import scipy.io
import scipy.optimize
import random
import time
import pandas as pd

#forcing funcion
def force(k,y):
    return -(1/4)*np.sin(k*y)

#deeponet solver (Using SPI Architecture)
class Deeponet_NavierStokes:
    def __init__(self,sensors,mu,nu,rho,dataste,dataset_bdr,dataset_ini,batch_size,batch_size_bdr,batch_size_ini,total_epochs,loss_steps,print_loss):
        self.sensors=sensors
        self.mu=mu
        self.nu=nu
        self.rho=rho
        self.dataset=dataset
        self.dataset_bdr=dataset_bdr
        self.dataset_ini=dataset_ini
        # self.dataset_sol=dataset_sol
        self.batch_size=batch_size
        self.batch_size_bdr=batch_size_bdr
        self.batch_size_ini=batch_size_ini
        self.total_epochs=total_epochs
        self.loss_total=[]
        self.loss_eqn=[]
        self.loss_b=[]
        self.loss_in=[]
        self.loss_sol=[]
        self.epochs=[]
        self.loss_steps=loss_steps
        self.model=self.build_model()
        self.print_loss=print_loss

    def build_model(self):
        optimizer = tf.keras.optimizers.Adam()
        tf.keras.backend.set_floatx('float32')

        #model for deepOnet (SPI) ################################

        #no of sensor points
        m=self.sensors
        branch_input = Input(shape=(m,), name='vector_input')
        trunck_inputx = Input(shape=(1,), name='x_input')
        trunck_inputy = Input(shape=(1,), name='y_input')
        trunck_inputt = Input(shape=(1,), name='t_input')

        #branch network
        branch_net = Dense(550, activation='tanh')(branch_input)
        branch_net = Dense(550, activation='tanh')(branch_net)
        branch_net = Dense(550, activation='tanh')(branch_net)
        branch_net = Dense(550, activation='tanh')(branch_net)
        branch_net = Dense(550, activation='tanh')(branch_net)
        branch_net = Dense(550, activation='tanh')(branch_net)
        branch_net = Dense(m, activation='tanh')(branch_net)

        # Trunck network
        trunck_netx = Dense(150, activation='tanh')(trunck_inputx)
        trunck_netx = Dense(150, activation='tanh')(trunck_netx)
        trunck_netx = Dense(150, activation='tanh')(trunck_netx)
        trunck_netx = Dense(150, activation='tanh')(trunck_netx)
        trunck_netx = Dense(150, activation='tanh')(trunck_netx)
        trunck_netx = Dense(150, activation='tanh')(trunck_netx)
        trunck_netx = Dense(m, activation='tanh')(trunck_netx)

        trunck_nety = Dense(150, activation='tanh')(trunck_inputy)
        trunck_nety = Dense(150, activation='tanh')(trunck_nety)
        trunck_nety = Dense(150, activation='tanh')(trunck_nety)
        trunck_nety = Dense(150, activation='tanh')(trunck_nety)
        trunck_nety = Dense(150, activation='tanh')(trunck_nety)
        trunck_nety = Dense(150, activation='tanh')(trunck_nety)
        trunck_nety = Dense(m, activation='tanh')(trunck_nety)

        trunck_nett = Dense(150, activation='tanh')(trunck_inputt)
        trunck_nett = Dense(150, activation='tanh')(trunck_nett)
        trunck_nett = Dense(150, activation='tanh')(trunck_nett)
        trunck_nett = Dense(150, activation='tanh')(trunck_nett)
        trunck_nett = Dense(150, activation='tanh')(trunck_nett)
        trunck_nett = Dense(150, activation='tanh')(trunck_nett)
        trunck_nett = Dense(m, activation='tanh')(trunck_nett)

        bias1 = tf.Variable(0., shape=(), trainable=True)
        bias2 = tf.Variable(0., shape=(), trainable=True)

        branch_net_1,branch_net_2 =branch_net[:,:int (m/2)], branch_net[:,int (m/2):]
        trunck_net_1x, trunck_net_2x = trunck_netx[:, :int (m/2)], trunck_netx[:, int(m/2):]
        trunck_net_1y, trunck_net_2y = trunck_nety[:, :int (m/2)], trunck_nety[:, int(m/2):]
        trunck_net_1t, trunck_net_2t = trunck_nett[:, :int (m/2)], trunck_nett[:, int(m/2):]

        dot_product_1 = Lambda(lambda x: tf.reduce_sum(x[0] * x[1] *x[2] *x[3], axis=1, keepdims=True))([branch_net_1, trunck_net_1x,trunck_net_1y,trunck_net_1t])
        dot_product_2 = Lambda(lambda x: tf.reduce_sum(x[0] * x[1] *x[2] *x[3], axis=1, keepdims=True))([branch_net_2, trunck_net_2x,trunck_net_2y,trunck_net_2t])

        # Create the model
        return tf.keras.Model(inputs=[branch_input, trunck_inputx,trunck_inputy,trunck_inputt], outputs=[dot_product_1+bias1, dot_product_2+bias2])

    # equation loss function 
    def loss(self,X_br,X_tr,k):
        with tf.GradientTape(persistent=True) as tape:

            x,y,t=tf.unstack(X_tr,axis=1)
            tape.watch(x)
            tape.watch(y)
            tape.watch(t)

            psi,p= self.model([X_br,x,y,t], training=True)
            u=tape.gradient(psi, y)
            v=-tape.gradient(psi, x)

            u_t=tape.gradient(u,t)
            u_x=tape.gradient(u,x)
            u_y=tape.gradient(u,y)
            v_t=tape.gradient(v,t)
            v_x=tape.gradient(v,x)
            v_y=tape.gradient(v,y)

        u_xx=tape.gradient(u_x,x)
        u_yy=tape.gradient(u_y,y)
        v_xx=tape.gradient(v_x,x)
        v_yy=tape.gradient(v_y,y)
        p_x=tape.gradient(p,x)
        p_y=tape.gradient(p,y)

        f_u=u_t+(u*u_x + v*u_y)+p_x-self.nu*(u_xx+u_yy)+self.mu*u+(1/4)*tf.sin(k*y)
        f_v=v_t+(u*v_x + v*v_y)+p_y-self.nu*(v_xx+v_yy)+self.mu*v

        del tape
        loss=(tf.reduce_mean(tf.square(f_u))+tf.reduce_mean(tf.square(f_v)))/2

        return loss

    #initial loss 
    def loss_ini(self,X_br,X_tr,k):
        with tf.GradientTape(persistent=True) as tape:

            x,y,t=tf.unstack(X_tr,axis=1)
            tape.watch(x)
            tape.watch(y)
            tape.watch(t)

            psi,p= self.model([X_br, x,y,t], training=True)
            u=tape.gradient(psi, y)
            v=-tape.gradient(psi, x)

        loss=tf.reduce_mean(tf.square(u))+tf.reduce_mean(tf.square(v))
        del tape
        return loss/2
    
    #loss using data
    # def loss_data(self, X_br, X_tr, k, Y_out):
    #     with tf.GradientTape(persistent=True) as tape:
    #         x, y, t = tf.unstack(X_tr, axis=1)
    #         tape.watch(x)
    #         tape.watch(y)
    #         tape.watch(t)

    #         # Forward pass through the model
    #         psi, p = self.model([X_br, x,y,t], training=True)

    #         # Compute gradients for u and v
    #         u = tape.gradient(psi, y)
    #         v = -tape.gradient(psi, x)

    #         # Compute second-order gradients for ux, uy, vx, and vy
    #         u_x = tape.gradient(u, x)
    #         u_y = tape.gradient(u, y)
    #         v_x = tape.gradient(v, x)
    #         v_y = tape.gradient(v, y)

    #     # Unstack Y_out to get actual values
    #     u_sol, v_sol, ux_sol, uy_sol, vx_sol, vy_sol = tf.unstack(Y_out, axis=1)

    #     # Compute the loss
    #     u_loss = tf.reduce_mean(tf.square(u - u_sol))
    #     v_loss = tf.reduce_mean(tf.square(v - v_sol))
    #     ux_loss = tf.reduce_mean(tf.square(u_x - ux_sol))
    #     uy_loss = tf.reduce_mean(tf.square(u_y - uy_sol))
    #     vx_loss = tf.reduce_mean(tf.square(v_x - vx_sol))
    #     vy_loss = tf.reduce_mean(tf.square(v_y - vy_sol))

    #     total_loss = (u_loss + v_loss+ ux_loss + uy_loss + vx_loss + vy_loss) / 6.0

    #     del tape
    #     return total_loss , tf.reduce_mean(abs(u_sol-u)/u_sol)

    #loss funcions at boundary
    def loss_bdr(self,X_br_bottom,X_tr_bottom,X_br_top,X_tr_top,X_br_left,X_tr_left,X_br_right,X_tr_right,k):
        with tf.GradientTape(persistent=True) as tape:

            x_bottom,y_bottom,t_bottom=tf.unstack(X_tr_bottom,axis=1)
            tape.watch(x_bottom)
            tape.watch(y_bottom)
            tape.watch(t_bottom)

            x_top,y_top,t_top=tf.unstack(X_tr_top,axis=1)
            tape.watch(x_top)
            tape.watch(y_top)
            tape.watch(t_top)

            x_left,y_left,t_left=tf.unstack(X_tr_left,axis=1)
            tape.watch(x_left)
            tape.watch(y_left)
            tape.watch(t_left)

            x_right,y_right,t_right=tf.unstack(X_tr_right,axis=1)
            tape.watch(x_right)
            tape.watch(y_right)
            tape.watch(t_right)

            # u_bottom,v_bottom,p_bottom= self.model([X_br_bottom, tf.stack((x_bottom,y_bottom,t_bottom),axis=1)], training=True)
            psi_b,p_bottom= self.model([X_br_bottom, x_bottom,y_bottom,t_bottom],training=True)
            u_bottom=tape.gradient(psi_b, y_bottom)
            v_bottom=-tape.gradient(psi_b, x_bottom)

            psi_top,p_top= self.model([X_br_top, x_top,y_top,t_top], training=True)
            u_top=tape.gradient(psi_top, y_top)
            v_top=-tape.gradient(psi_top, x_top)

            psi_left,p_left= self.model([X_br_left,x_left,y_left,t_left], training=True)
            u_left=tape.gradient(psi_left, y_left)
            v_left=-tape.gradient(psi_left, x_left)

            psi_right,p_right= self.model([X_br_right,x_right,y_right,t_right], training=True)
            u_right=tape.gradient(psi_right, y_right)
            v_right=-tape.gradient(psi_right, x_right)

        u_x_bottom=tape.gradient(u_bottom,x_bottom)
        u_y_bottom=tape.gradient(u_bottom,y_bottom)
        v_x_bottom=tape.gradient(v_bottom,x_bottom)
        v_y_bottom=tape.gradient(v_bottom,y_bottom)

        u_x_top=tape.gradient(u_top,x_top)
        u_y_top=tape.gradient(u_top,y_top)
        v_x_top=tape.gradient(v_top,x_top)
        v_y_top=tape.gradient(v_top,y_top)

        u_x_left=tape.gradient(u_left,x_left)
        u_y_left=tape.gradient(u_left,y_left)
        v_x_left=tape.gradient(v_left,x_left)
        v_y_left=tape.gradient(v_left,y_left)

        u_x_right=tape.gradient(u_right,x_right)
        u_y_right=tape.gradient(u_right,y_right)
        v_x_right=tape.gradient(v_right,x_right)
        v_y_right=tape.gradient(v_right,y_right)

        loss1=tf.reduce_mean(tf.square(u_bottom-u_top))+tf.reduce_mean(tf.square(v_bottom-v_top))
        loss2=tf.reduce_mean(tf.square(u_left-u_right))+tf.reduce_mean(tf.square(v_left-v_right))

        loss3=tf.reduce_mean(tf.square(u_x_bottom-u_x_top))+tf.reduce_mean(tf.square(v_x_bottom-v_x_top))
        loss4=tf.reduce_mean(tf.square(u_x_left-u_x_right))+tf.reduce_mean(tf.square(v_x_left-v_x_right))
        loss5=tf.reduce_mean(tf.square(u_y_bottom-u_y_top))+tf.reduce_mean(tf.square(v_y_bottom-v_y_top))
        loss6=tf.reduce_mean(tf.square(u_y_left-u_y_right))+tf.reduce_mean(tf.square(v_y_left-v_y_right))


        del tape
        return (loss1+loss2+loss3+loss4+loss5+loss6)/12
    
    #training loop, customisable
    def train(self):
        initial_learning_rate = 1e-3

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps=20, decay_rate=0.9)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        time_start = time.time()

        for epoch in range(self.total_epochs):
            # Sample data from the main dataset
            sampled_indices = random.sample(range(len(self.dataset[0])), min(100, len(self.dataset[0])))
            sampled_X_br = [self.dataset[0][i] for i in sampled_indices]
            sampled_X_tr = [self.dataset[1][i] for i in sampled_indices]
            sampled_k = [self.dataset[2][i] for i in sampled_indices]

            # Sample data from the initial dataset
            sampled_indices_ini = random.sample(range(len(self.dataset_ini[0])), min(100, len(self.dataset_ini[0])))
            sampled_X_br_ini = [self.dataset_ini[0][i] for i in sampled_indices_ini]
            sampled_X_tr_ini = [self.dataset_ini[1][i] for i in sampled_indices_ini]
            sampled_k_ini = [self.dataset_ini[2][i] for i in sampled_indices_ini]

            # Sample data from the boundary dataset
            sampled_indices_bdr = random.sample(range(len(self.dataset_bdr[0])), min(100, len(self.dataset_bdr[0])))
            sampled_X_br_bottom = [self.dataset_bdr[0][i] for i in sampled_indices_bdr]
            sampled_X_tr_bottom = [self.dataset_bdr[1][i] for i in sampled_indices_bdr]
            sampled_X_br_top = [self.dataset_bdr[2][i] for i in sampled_indices_bdr]
            sampled_X_tr_top = [self.dataset_bdr[3][i] for i in sampled_indices_bdr]
            sampled_X_br_left = [self.dataset_bdr[4][i] for i in sampled_indices_bdr]
            sampled_X_tr_left = [self.dataset_bdr[5][i] for i in sampled_indices_bdr]
            sampled_X_br_right = [self.dataset_bdr[6][i] for i in sampled_indices_bdr]
            sampled_X_tr_right = [self.dataset_bdr[7][i] for i in sampled_indices_bdr]
            sampled_k_bdr = [self.dataset_bdr[8][i] for i in sampled_indices_bdr]

            # # Sample data from the solution dataset
            # sampled_indices_sol = random.sample(range(len(self.dataset_sol[0])), min(3000, len(self.dataset_sol[0])))
            # sampled_X_br_sol = [self.dataset_sol[0][i] for i in sampled_indices_sol]
            # sampled_X_tr_sol = [self.dataset_sol[1][i] for i in sampled_indices_sol]
            # sampled_k_sol = [self.dataset_sol[2][i] for i in sampled_indices_sol]
            # sampled_Y_out_sol = [self.dataset_sol[3][i] for i in sampled_indices_sol]

            # Convert to tensors
            sampled_X_br = tf.convert_to_tensor(sampled_X_br, dtype=tf.float32)
            sampled_X_tr = tf.convert_to_tensor(sampled_X_tr, dtype=tf.float32)
            sampled_k = tf.convert_to_tensor(sampled_k, dtype=tf.float32)

            sampled_X_br_ini = tf.convert_to_tensor(sampled_X_br_ini, dtype=tf.float32)
            sampled_X_tr_ini = tf.convert_to_tensor(sampled_X_tr_ini, dtype=tf.float32)
            sampled_k_ini = tf.convert_to_tensor(sampled_k_ini, dtype=tf.float32)

            sampled_X_br_bottom = tf.convert_to_tensor(sampled_X_br_bottom, dtype=tf.float32)
            sampled_X_tr_bottom = tf.convert_to_tensor(sampled_X_tr_bottom, dtype=tf.float32)
            sampled_X_br_top = tf.convert_to_tensor(sampled_X_br_top, dtype=tf.float32)
            sampled_X_tr_top = tf.convert_to_tensor(sampled_X_tr_top, dtype=tf.float32)
            sampled_X_br_left = tf.convert_to_tensor(sampled_X_br_left, dtype=tf.float32)
            sampled_X_tr_left = tf.convert_to_tensor(sampled_X_tr_left, dtype=tf.float32)
            sampled_X_br_right = tf.convert_to_tensor(sampled_X_br_right, dtype=tf.float32)
            sampled_X_tr_right = tf.convert_to_tensor(sampled_X_tr_right, dtype=tf.float32)
            sampled_k_bdr = tf.convert_to_tensor(sampled_k_bdr, dtype=tf.float32)

            # sampled_X_br_sol = tf.convert_to_tensor(sampled_X_br_sol, dtype=tf.float32)
            # sampled_X_tr_sol = tf.convert_to_tensor(sampled_X_tr_sol, dtype=tf.float32)
            # sampled_k_sol = tf.convert_to_tensor(sampled_k_sol, dtype=tf.float32)
            # sampled_Y_out_sol = tf.convert_to_tensor(sampled_Y_out_sol, dtype=tf.float32)

            # Training on the main dataset
            with tf.GradientTape() as tape:
                loss_value = self.loss(sampled_X_br, sampled_X_tr, sampled_k)
            grads = tape.gradient(loss_value, self.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            del tape

            # Training on the initial dataset
            if(epoch%10==0):
                with tf.GradientTape() as tape:
                    loss_value_ini = self.loss_ini(sampled_X_br_ini, sampled_X_tr_ini, sampled_k_ini)
                grads = tape.gradient(loss_value_ini, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                del tape

            # Training on the boundary dataset
            if(epoch%10==0):
                with tf.GradientTape() as tape:
                    loss_value_bdr = self.loss_bdr(
                        sampled_X_br_bottom, sampled_X_tr_bottom,
                        sampled_X_br_top, sampled_X_tr_top,
                        sampled_X_br_left, sampled_X_tr_left,
                        sampled_X_br_right, sampled_X_tr_right,
                        sampled_k_bdr)
                grads = tape.gradient(loss_value_bdr, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                del tape

            # Training on the solution dataset
            # with tf.GradientTape() as tape:
            #     loss_value_sol, rel_loss = self.loss_data(sampled_X_br_sol, sampled_X_tr_sol, sampled_k_sol, sampled_Y_out_sol)
            # grads = tape.gradient(loss_value_sol, self.model.trainable_variables)
            # optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            # del tape

            # Logging losses
            if epoch % 10 == 0:
                loss_total_f = (loss_value + loss_value_bdr + loss_value_ini ) / 3
                self.loss_total.append(loss_total_f)
                self.loss_eqn.append(loss_value)
                self.loss_in.append(loss_value_ini)
                self.loss_b.append(loss_value_bdr)
                # self.loss_sol.append(loss_value_sol)
                self.epochs.append(epoch)

            else:
                loss_total_f = (loss_value)
                self.loss_total.append(loss_total_f)
                self.loss_eqn.append(loss_value)
                # self.loss_sol.append(loss_value_sol)
                self.epochs.append(epoch)
            # Print loss periodically
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, eqn loss={loss_value:.4e}, bdr loss={loss_value_bdr:.4e}, initial loss={loss_value_ini:.4e}")

            else:
                print(f"Epoch {epoch}, eqn loss={loss_value:.4e}")

        time_end = time.time()
        span = time_end - time_start
        print("Time per epoch: ", span / self.total_epochs)
        print("Total time: ", span)

    #solution prediction
    def solution(self,X_br,X_tr):
        with tf.GradientTape(persistent=True) as tape:

            x,y,t=tf.unstack(X_tr,axis=1)
            tape.watch(x)
            tape.watch(y)
            tape.watch(t)

            psi,p= self.model([X_br, x,y,t])
            u=tape.gradient(psi, y)
            v=-tape.gradient(psi, x)

            u_t=tape.gradient(u,t)
            u_x=tape.gradient(u,x)
            u_y=tape.gradient(u,y)
            v_t=tape.gradient(v,t)
            v_x=tape.gradient(v,x)
            v_y=tape.gradient(v,y)

        u_xx=tape.gradient(u_x,x)
        u_yy=tape.gradient(u_y,y)

        v_xx=tape.gradient(v_x,x)
        v_yy=tape.gradient(v_y,y)
        p_x=tape.gradient(p,x)
        p_y=tape.gradient(p,y)

        del tape
        return np.array(p),np.array(u),np.array(v),np.array(u_t),np.array(u_x),np.array(u_y),np.array(u_xx),np.array(u_yy),\
               np.array(v_t),np.array(v_x),np.array(v_y),np.array(v_xx),np.array(v_yy),np.array(p_x),np.array(p_y)

'solution Data '
# from google.colab import drive
# drive.mount('/content/drive')
# file_path = '/content/drive/MyDrive/Low_Re_sol'

# data=scipy.io.loadmat(file_path)

# # Extract the necessary variables
# U = data['U_spectral']
# V = data['V_spectral']
# Ux = data['Ux_spectral']
# Uy = data['Uy_spectral']
# Vx = data['Vx_spectral']
# Vy = data['Vy_spectral']
# Time = data['Time']

# # Create a meshgrid for x and y coordinates
# x = np.linspace(0, 2*np.pi, 512)
# y = np.linspace(0, 2*np.pi, 512)
# x_grid, y_grid = np.meshgrid(x, y)

# # Set k=4 and generate the data
# k = 1

# data_list = []

# step_size = 20  # Sampling every 10th point

# for t_idx in range(len(Time)):
#     t_value = Time[t_idx][0]

#     for i in range(0,512,step_size):
#         for j in range(0,512,step_size):
#             row = [
#                 x_grid[i, j],  # x
#                 y_grid[i, j],  # y
#                 t_value,       # t
#                 k,             # k
#                 U[t_idx, i, j],  # u
#                 V[t_idx, i, j],  # v
#                 Ux[t_idx, i, j], # ux
#                 Uy[t_idx, i, j], # uy
#                 Vx[t_idx, i, j], # vx
#                 Vy[t_idx, i, j]  # vy
#             ]
#             data_list.append(row)

# columns = ['x', 'y', 't', 'k', 'u', 'v', 'ux', 'uy', 'vx', 'vy']

# df = pd.DataFrame(data_list, columns=columns)

# # Inspect the DataFrame
# print(df.head())


"""# continuation"""
outputfile='deeponet.mat'                 # output filename

n_func=1                                  #no of forcing functions
sensors=500                               #no of sensor points to represent the forcing function
n_bdr=20                                  #no of boundary points at each boundary
spacial_resol=128                         #spacial resolution
temporal_resol= 50                        #temporal resolution
n_ini=spacial_resol*spacial_resol         #no of initial points
n_train=(spacial_resol**2)*temporal_resol #total no of equation training points

x0=0.0                                    #x domain start
xl=2*np.pi                                #x domain end

y0=0.0                                    #y domain left
yl=2*np.pi                                #y domain right

t0=0.0                                    #start time
tl=10.0                                   #end time

#constants
mu=0.1
nu=0.001                                  #coeff of viscosity
rho=1                                     #density

Re=rho*xl/mu
Re

def create_domain(x_range, y_range, t_range, sresol,tresol):
    return np.linspace(x_range[0], x_range[1], sresol), np.linspace(y_range[0], y_range[1], sresol), np.linspace(t_range[0], t_range[1], tresol)

def create_boundary(x_range, y_range, t_length, n_points):
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    t = np.random.rand(n_points)
    return x, y, t

def meshgrid_and_flatten(x, y, t):
    xx, yy, tt = np.meshgrid(x, y, t, indexing='ij')
    return np.reshape(xx, -1), np.reshape(yy, -1), np.reshape(tt, -1)

def concatenate_boundary(x, y, t, position):
    if position == 'bottom' or position == 'top':
        y = np.zeros(len(x)) if position == 'bottom' else yl * np.ones(len(x))
    elif position == 'left' or position == 'right':
        x = np.zeros(len(y)) if position == 'left' else xl * np.ones(len(y))
    return np.concatenate([x[:, None], y[:, None], t[:, None]], axis=1)

# Create domain and boundaries
x, y, t = create_domain((x0, xl), (y0, yl), (t0, tl), spacial_resol,temporal_resol)
x_bdr, y_bdr, t_bdr = create_boundary((x0, xl), (y0, yl), tl, n_bdr)

# Meshgrids for computation
xx, yy, tt = meshgrid_and_flatten(x, y, t)
t_ini = np.zeros(n_ini)
x_ini, y_ini = np.meshgrid(x, y, indexing='ij')
x_ini, y_ini, t_ini = x_ini.flatten(), y_ini.flatten(), t_ini.flatten()

# Concatenate points for various boundaries
Y = np.stack([xx, yy, tt], axis=-1)
Y_bottom = concatenate_boundary(x_bdr, y0, t_bdr, 'bottom')
Y_top = concatenate_boundary(x_bdr, yl, t_bdr, 'top')
Y_left = concatenate_boundary(x0, y_bdr, t_bdr, 'left')
Y_right = concatenate_boundary(xl, y_bdr, t_bdr, 'right')
Y_ini = np.stack([x_ini, y_ini, t_ini], axis=-1)

#sensor points
y_sensor=np.linspace(y0,yl,sensors)

#forcing function wave numbers
k=np.array([4])

#stores the forcinf function values at sensor points
f=[]
for i in range(n_func):
    f.append(force(k[i],y_sensor))
f=np.array(f)
f.shape

'solution Data part'
# X_data_branch=[]
# X_data_trunck=[]
# k_data=[]
# Y_out=[]

# for i in range(len(df)):
#   X_data_branch.append(force(df['k'][i],y_sensor))
#   X_data_trunck.append(np.array([df['x'][i],df['y'][i],df['t'][i]]))
#   k_data.append(df['k'][i])
#   Y_out.append(np.array([df['u'][i],df['v'][i],df['ux'][i],df['uy'][i],df['vx'][i],df['vy'][i]]))

# X_data_branch = np.array(X_data_branch)
# X_data_trunck = np.array(X_data_trunck)
# k_data = np.array(k_data)
# Y_out = np.array(Y_out)

X_train_branch=[]
X_train_trunck=[]
k_train=[]

for i in range(n_func):
    for j in range(n_train):
        X_train_branch.append(f[i])
        X_train_trunck.append(Y[j])
        k_train.append(k[i])

X_train_branch=np.array(X_train_branch)
X_train_trunck=np.array(X_train_trunck)
k_train=np.array(k_train)

#boundary training data

k_train_bdr=[]
k_train_ini=[]
X_train_branch_bottom=[]
X_train_branch_top=[]
X_train_branch_left=[]
X_train_branch_right=[]
X_train_branch_ini=[]

X_train_trunck_bottom=[]
X_train_trunck_top=[]
X_train_trunck_left=[]
X_train_trunck_right=[]
X_train_trunck_ini=[]

for i in range(n_func):
    for j in range(n_bdr):
        X_train_branch_bottom.append(f[i])
        X_train_trunck_bottom.append(Y_bottom[j])

        X_train_branch_top.append(f[i])
        X_train_trunck_top.append(Y_top[j])

        X_train_branch_left.append(f[i])
        X_train_trunck_left.append(Y_left[j])

        X_train_branch_right.append(f[i])
        X_train_trunck_right.append(Y_right[j])

        k_train_bdr.append(k[i])


#initial data
for i in range(n_func):
    for j in range(n_ini):

        X_train_branch_ini.append(f[i])
        X_train_trunck_ini.append(Y_ini[j])
        k_train_ini.append(k[i])


# List of variables to be converted to NumPy arrays
variables = [
    'k_train_bdr', 'k_train_ini',
    'X_train_branch_bottom', 'X_train_branch_top', 'X_train_branch_right', 'X_train_branch_left', 'X_train_branch_ini',
    'X_train_trunck_bottom', 'X_train_trunck_top', 'X_train_trunck_right', 'X_train_trunck_left', 'X_train_trunck_ini'
]

# Convert each list in the variables list to a NumPy array
for var in variables:
    globals()[var] = np.array(globals()[var])

def to_tensor(x):
    return tf.convert_to_tensor(x,dtype=tf.float32, name=None)

variables = {
    'X_train_br': X_train_branch,
    'X_train_tr': X_train_trunck,
    'k_train': k_train,
    'X_train_br_bottom': X_train_branch_bottom,
    'X_train_tr_bottom': X_train_trunck_bottom,
    'X_train_br_top': X_train_branch_top,
    'X_train_tr_top': X_train_trunck_top,
    'X_train_br_left': X_train_branch_left,
    'X_train_tr_left': X_train_trunck_left,
    'X_train_br_right': X_train_branch_right,
    'X_train_tr_right': X_train_trunck_right,
    'X_train_br_ini': X_train_branch_ini,
    'X_train_tr_ini': X_train_trunck_ini,
    'k_train_bdr': k_train_bdr,
    'k_train_ini': k_train_ini
}

# Convert all variables to tensors
for key, value in variables.items():
    globals()[key] = to_tensor(value)


# Convert NumPy arrays to TensorFlow tensors
def to_tensor(x):
    return tf.convert_to_tensor(x, dtype=tf.float32)

# # Create a dictionary of the variables
# variables = {
#     'X_data_branch': X_data_branch,
#     'X_data_trunck': X_data_trunck,
#     'k_data': k_data,
#     'Y_out': Y_out
# }

# # Convert all variables to tensors
# for key, value in variables.items():
#     variables[key] = to_tensor(value)

# # Assign tensors to global variables
# X_data_branch = variables['X_data_branch']
# X_data_trunck = variables['X_data_trunck']
# k_data = variables['k_data']
# Y_out = variables['Y_out']

'Final datasets for training'
dataset = [X_train_br, X_train_tr, k_train]

dataset_bdr= [X_train_br_bottom, X_train_tr_bottom,
                                                 X_train_br_top, X_train_tr_top,
                                                 X_train_br_left, X_train_tr_left,
                                                 X_train_br_right, X_train_tr_right,
                                                 k_train_bdr]

dataset_ini =[X_train_br_ini, X_train_tr_ini, k_train_ini]
# dataset_sol=[X_data_branch,X_data_trunck,k_data,Y_out]

batch_size = 2000
batch_size_bdr=100
batch_size_ini=100
total_epochs=2000
loss_save_steps=1
loss_print_steps=1

deeponet=Deeponet_NavierStokes(sensors,mu,nu,rho,dataset,dataset_bdr,dataset_ini,batch_size,batch_size_bdr,batch_size_ini,total_epochs,loss_save_steps,loss_print_steps)

'Train the model'
deeponet.train()

'Saving the trained model'
deeponet.model.save('deeponet_Navierstokes.keras')