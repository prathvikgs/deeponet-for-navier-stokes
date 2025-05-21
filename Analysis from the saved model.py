import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

'use the saved model'
model = tf.keras.models.load_model('my_model.keras')


n_func=1                                  #no of forcing functions
sensors=100                               #no of sensor points to represent the forcing function
n_bdr=200                                 #no of boundary points at each boundary
spacial_resol=512                         #spacial resolution
temporal_resol=3                          #temporal resolution
n_ini=spacial_resol*spacial_resol         #no of initial points
n_train=(spacial_resol**2)*temporal_resol #total no of equation training points

x0=0.0                                    #x domain start
xl=2*np.pi                                #x domain end

y0=0.0                                    #y domain left
yl=2*np.pi                                #y domain right
 
t0=0.0                                    #start time
tl=10.0                                   #end time

#constants
mu=0.05
nu=0.001                                  #coeff of viscosity
rho=1   

def create_domain(x_range, y_range, t_range, resol):
    return np.linspace(x_range[0], x_range[1], resol), np.linspace(y_range[0], y_range[1], resol), np.linspace(t_range[0], t_range[1], resol)

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

def force(k,y):
    return -(1/4)*np.sin(k*y)
# Create domain and boundaries
x, y, t = create_domain((x0, xl), (y0, yl), (t0, tl), spacial_resol)
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


#domain for trunck net
x=np.linspace(x0,xl,spacial_resol)
y=np.linspace(y0,yl,spacial_resol)
t=10*np.ones(spacial_resol*spacial_resol)

xx,yy=np.meshgrid(x,y,indexing='ij')
xx=np.reshape(xx,-1)
yy=np.reshape(yy,-1)
tt=np.reshape(t,-1)
 
Y=np.concatenate([xx[:,None],yy[:,None],tt[:,None]],axis=1) #interior points
k_new=4
X_test_branch=[]
X_test_trunck=[]

#new forcing function to compare
f_new=force(k_new,y_sensor)
for i in range(1):
    for j in range(spacial_resol*spacial_resol*1):
        X_test_branch.append(f_new)
        X_test_trunck.append(Y[j])

        
#testing data
X_test_branch1=np.array(X_test_branch)
X_test_trunck1=np.array(X_test_trunck)

X_test_branch=tf.convert_to_tensor(X_test_branch1, dtype=tf.float32, dtype_hint=None, name=None)
X_test_trunck=tf.convert_to_tensor(X_test_trunck1, dtype=tf.float32, dtype_hint=None, name=None)


def solution(X_br,X_tr):
    with tf.GradientTape(persistent=True) as tape:

        x,y,t=tf.unstack(X_tr,axis=1)
        tape.watch(x)
        tape.watch(y)
        tape.watch(t)

        u,v,p= model([X_br, tf.stack((x,y,t),axis=1)])

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
    return np.array(p),np.array(u),np.array(v),np.array(u_t),np.array(u_x),np.array(u_y),np.array(u_xx),np.array(u_yy),np.array(v_t),np.array(v_x),np.array(v_y),np.array(v_xx),np.array(v_yy),np.array(p_x),np.array(p_y)


'get all the outputs'
p,u,v,u_t,u_x,u_y,u_xx,u_yy,v_t,v_x,v_y,v_xx,v_yy,p_x,p_y=solution(X_test_branch,X_test_trunck) 

mm=np.reshape(u,(spacial_resol,spacial_resol))
x = np.linspace(0, 2*np.pi, spacial_resol)
y = np.linspace(0, 2*np.pi, spacial_resol)
X,Y=np.meshgrid(x,y)

c = plt.pcolor(X, Y, mm, cmap='viridis')  # Apply a colormap
plt.colorbar(c)  # Show color scale

plt.xlabel('X coordinate')  # X-axis label
plt.ylabel('Y coordinate')  # Y-axis label

# Show the plot
plt.show()