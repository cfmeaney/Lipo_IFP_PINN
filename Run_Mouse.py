################################################################################
### PINN for Identification of Pressure from Liposome Data (Synthetic Data)
################################################################################

# Get rid of warning messages and CUDA loadings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

# Import Packages
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io
import matplotlib.animation as animation
from matplotlib import cm
import time
from os import environ
import tensorflow_probability as tfp
from scipy import ndimage

print(tf.__version__)

cwdir=os.getcwd()
print(cwdir)

# Function for L-BFGS optimization
def function_factory(model, loss, train_x, train_y):
    """A factory to create a function required by tfp.optimizer.lbfgs_minimize.

    Args:
        model [in]: an instance of `tf.keras.Model` or its subclasses.
        loss [in]: a function with signature loss_value = loss(pred_y, true_y).
        train_x [in]: the input part of training data.
        train_y [in]: the output part of training data.

    Returns:
        A function that has a signature of:
            loss_value, gradients = f(model_parameters).
    """

    # obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model.trainable_variables)
    n_tensors = len(shapes)

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
    # prepare required information first
    count = 0
    idx = [] # stitch indices
    part = [] # partition indices

    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
        part.extend([i]*n)
        count += n

    part = tf.constant(part)

    @tf.function
    def assign_new_model_parameters(params_1d):
        """A function updating the model's parameters with a 1D tf.Tensor.

        Args:
            params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
        """

        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            model.trainable_variables[i].assign(tf.reshape(param, shape))

    # now create a function that will be returned by this factory
    @tf.function
    def f(params_1d):
        """A function that can be used by tfp.optimizer.lbfgs_minimize.

        This function is created by function_factory.

        Args:
           params_1d [in]: a 1D tf.Tensor.

        Returns:
            A scalar loss and the gradients w.r.t. the `params_1d`.
        """

        # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
        with tf.GradientTape() as tape:
            # update the parameters in the model
            assign_new_model_parameters(params_1d)
            # calculate the loss
            loss_value = model(train_x, training=True)

        # calculate gradients and convert to 1D tf.Tensor
        grads = tape.gradient(loss_value, model.trainable_variables)
        grads = tf.dynamic_stitch(idx, grads)

        # print out iteration & loss
        f.iter.assign_add(1)
        tf.print("Iter:", f.iter, "loss:", loss_value)

        # store loss value so we can retrieve later
        tf.py_function(f.history.append, inp=[loss_value], Tout=[])

        return loss_value, grads

    # store these information as members so we can use them outside the scope
    f.iter = tf.Variable(0)
    f.idx = idx
    f.part = part
    f.shapes = shapes
    f.assign_new_model_parameters = assign_new_model_parameters
    f.history = []

    return f

# Start Timing Code
t0 = time.time()

# Accessible Definitions
path = f'{cwdir}/Data_Inputs.mat' # Location of FEniCS data
Adam_Epochs = 300000 # Number of epochs of Adam optimization
LBFGS_Epochs = 50000 # Number of epochs of L-BFGS optimization
q = 100 # Number of RK time steps (max 500)

# Load in RK weights
IRK_weights = np.float32(np.loadtxt(f'{cwdir}/IRK_weights/Butcher_IRK' + str(q) + '.txt', ndmin=2))
IRK_times = IRK_weights[q**2+q:]
IRK_weights = IRK_weights[:q**2+q].reshape((q+1,q))
IRK_alpha = tf.constant(IRK_weights[:-1,:], dtype='float32')
IRK_beta = tf.constant(IRK_weights[-1:,:], dtype='float32')

# Set random seeds
np.random.seed(20515824)
tf.random.set_seed(20515824)

# On/offs for code grouping
define_parameters = 'on'
load_data = 'on'
nondimensionalize = 'on'
data_preprocessing = 'on'
define_model = 'on'
train_model = 'on'
printing = 'on'
saving = 'on'

script_path = os.path.abspath(__file__)
script_filename = os.path.basename(script_path)
subject = '01'
input_file = 'Mouse_Inputs/Mouse_' + subject + '_Inputs.mat'
output_file = 'Mouse_Results/Data_Results_' + subject + '.mat'

if define_parameters == 'on':
    # Define parameters
    P_v = 25.0 # (mmHg)
    L_pS_over_V_normal = 2.52e-6 # (1/mmHg*s)
    L_pS_over_V_tumour = 2.6e-6 # (1/mmHg*s)
    sigma_normal = 1.0 # (1)
    sigma_tumour = 0.2 # (1)
    f_normal = 1.0 # (1)
    f_tumour = 0.5 # (1)
    K_normal = 8.53e-8 # (cm^2/mmHg*s)
    K_tumour = 4.13e-8 # (cm^2/mmHg*s)
    k_d = 5.1e-6 # (1/s)
    # Define paramter grouping
    mu_normal = L_pS_over_V_normal*(1-sigma_normal)
    mu_tumour = L_pS_over_V_tumour*(1-sigma_tumour)
    eta_normal = f_normal*K_normal
    eta_tumour = f_tumour*K_tumour

if load_data == 'on':
    # load in FEniCS data
    data = scipy.io.loadmat(input_file)
    # Extract position
    xyz = data['x'][0][-1]*0.1 # in cm
    # Extract time
    dt = data['dt'][0][-1]*60*60 # in seconds
    # Extract initial and final solution
    u = data['u'][0]
    u1 = u[-1]
    u0 = np.zeros(u1.shape)
    data_shape = u0.shape
    u0 = u0.flatten()
    u1 = u1.flatten()
    # Define spatial differentials
    dx = xyz[-1,0]/data_shape[0]
    dy = dx # xyz[-1,1]/data_shape[1]
    dz = dx # xyz[-1,2]/data_shape[2]
    # Extract tumour area
    tumour_area = data['tumour_area'][0][-1].astype('float32')
    # Calculate plasma concentration
    C_p_params = data['C_p_params'][0]
    a = C_p_params[0]
    b = C_p_params[1]
    times_for_C_p = IRK_times*dt/60/60
    C_p_t = a*np.exp(-b*times_for_C_p)
    C_p_t_list = []
    for i in range(C_p_t.shape[0]):
        C_p_t_list.append(C_p_t[i]*np.ones(tumour_area.shape).flatten())

if nondimensionalize == 'on':
    """
    Nondimensionalize using scales of T = 1/P_v*mu_tumour*a, X = 1/sqrt(mu_tumour*a/eta_tumour), P = P_v
    """
    # Scale mu_tumour by plasma magnitude
    # mu_tumour = mu_tumour/a
    # Use tumour area to create spatially-dependent parameters
    mu = (1 - mu_normal/mu_tumour)*tumour_area + mu_normal/mu_tumour
    eta = (1 - eta_normal/eta_tumour)*tumour_area + eta_normal/eta_tumour
    k_d_nd = k_d/(P_v*mu_tumour*a)
    # Apply a mean filter to the mu and eta parameter arrays
    n = 3
    mu = ndimage.convolve(mu, np.full((n, n, n), 1/n**3))
    eta = ndimage.convolve(eta, np.full((n, n, n), 1/n**3))
    # Rescale t and x inputs
    dt = dt*P_v*mu_tumour*a
    xyz = xyz*np.sqrt(mu_tumour*a/eta_tumour)
    dx = dx*np.sqrt(mu_tumour*a/eta_tumour)
    dy = dy*np.sqrt(mu_tumour*a/eta_tumour)
    dz = dz*np.sqrt(mu_tumour*a/eta_tumour)
    # Rescale the plasma concentration
    C_p_t_list = [C_p_t/a for C_p_t in C_p_t_list]

if data_preprocessing == 'on':
    # Numerically differentiate eta
    grad_eta = np.gradient(eta,dx,dy,dz)
    eta_x = grad_eta[0]
    eta_y = grad_eta[1]
    eta_z = grad_eta[2]
    # Flatten and reshape mu and eta for PDE loss
    mu = mu.flatten()
    eta = eta.flatten()
    eta_x = eta_x.flatten()
    eta_y = eta_y.flatten()
    eta_z = eta_z.flatten()
    mu = tf.reshape(tf.stack(q*[mu], axis=1), [mu.shape[0], q])
    eta = tf.reshape(tf.stack(q*[eta], axis=1), [eta.shape[0], q])
    eta_x = tf.reshape(tf.stack(q*[eta_x], axis=1), [eta_x.shape[0], q])
    eta_y = tf.reshape(tf.stack(q*[eta_y], axis=1), [eta_y.shape[0], q])
    eta_z = tf.reshape(tf.stack(q*[eta_z], axis=1), [eta_z.shape[0], q])
    tumour_area = tumour_area.flatten()
    # Define network inputs (x)
    train_inputs = xyz
    input_min = xyz.min(0)
    input_max = xyz.max(0)
    train_inputs = 2*(train_inputs - input_min)/(input_max - input_min) - 1
    # Define snapshot outputs (u0 and u1)
    output_max = max(np.amax(u0), np.amax(u1))
    train_outputs_t0 = u0/output_max # Initial snapshot solution data
    train_outputs_t1 = u1/output_max # Final snapshot solution data
    # Format the exact snapshots for the loss function
    U0 = q*[train_outputs_t0]
    U0 = tf.stack(U0, axis=1)
    U0 = tf.cast(U0, tf.float32)
    U1 = q*[train_outputs_t1]
    U1 = tf.stack(U1, axis=1)
    U1 = tf.cast(U1, tf.float32)
    tumour_area = q*[tumour_area]
    tumour_area = tf.stack(tumour_area, axis=1)
    tumour_area = tf.cast(tumour_area, tf.float32)
    C_p = tf.stack(C_p_t_list, axis=1)
    C_p = tf.cast(C_p, tf.float32)

if define_model == 'on':

    # Define model architecture
    Inputs = tf.keras.layers.Input(shape=(3,))
    Dense_1 = tf.keras.layers.Dense(50, activation='tanh', kernel_initializer="glorot_normal")(Inputs)
    Dense_2 = tf.keras.layers.Dense(50, activation='tanh', kernel_initializer="glorot_normal")(Dense_1)
    Dense_3 = tf.keras.layers.Dense(50, activation='tanh', kernel_initializer="glorot_normal")(Dense_2)
    Dense_4 = tf.keras.layers.Dense(50, activation='tanh', kernel_initializer="glorot_normal")(Dense_3)
    Prediction = tf.keras.layers.Dense(q+1)(Dense_4)

    # Define prediction network
    model_prediction = tf.keras.models.Model(inputs=Inputs, outputs=Prediction)

    # Define TF variables for autodifferentiation
    train_inputs = train_inputs.astype(np.float32)
    train_inputs_var = tf.Variable(train_inputs, name='train_inputs_var')
    dummy = tf.ones([train_inputs_var.shape[0], q], dtype=np.float32)
    dummy_P = tf.ones([train_inputs_var.shape[0], 1], dtype=np.float32)

    # Define loss function layer
    class Loss_Layer(tf.keras.layers.Layer):
        def __init__(self):
            super(Loss_Layer, self).__init__()

        # Define custom loss function
        def custom_loss(self, pred):
            # Setep tape and perform derivatives
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(train_inputs_var)
                tape.watch(dummy)
                tape.watch(dummy_P)
                # Get k and P from model
                k_and_P = model_prediction(train_inputs_var)
                k = k_and_P[:,:-1]
                P = k_and_P[:,-1]
                P = tf.reshape(P, [P.shape[0], 1])
                P = 0.5*(tf.tanh(P) + 1)
                # Perform the first derivative of k
                g_U = tape.gradient(k, train_inputs_var, output_gradients=dummy)
                g_Ux = g_U[:,0]
                g_Uy = g_U[:,1]
                g_Uz = g_U[:,2]
                k_x = tape.gradient(g_Ux, dummy)
                k_y = tape.gradient(g_Uy, dummy)
                k_z = tape.gradient(g_Uz, dummy)
                g_Ux_U = tape.gradient(k_x, train_inputs_var, output_gradients=dummy)
                g_Uy_U = tape.gradient(k_y, train_inputs_var, output_gradients=dummy)
                g_Uz_U = tape.gradient(k_z, train_inputs_var, output_gradients=dummy)
                g_Ux_Ux = g_Ux_U[:,0]
                g_Uy_Uy = g_Uy_U[:,1]
                g_Uz_Uz = g_Uz_U[:,2]
                # Perform the first derivative of P
                g_P = tape.gradient(P, train_inputs_var, output_gradients=dummy_P)
                g_Px = g_P[:,0]
                g_Py = g_P[:,1]
                g_Pz = g_P[:,2]
                P_x = tape.gradient(g_Px, dummy_P)
                P_y = tape.gradient(g_Py, dummy_P)
                P_z = tape.gradient(g_Pz, dummy_P)
                g_Px_P = tape.gradient(P_x, train_inputs_var, output_gradients=dummy_P)
                g_Py_P = tape.gradient(P_y, train_inputs_var, output_gradients=dummy_P)
                g_Pz_P = tape.gradient(P_z, train_inputs_var, output_gradients=dummy_P)
                g_Px_Px = g_Px_P[:,0]
                g_Py_Py = g_Py_P[:,1]
                g_Pz_Pz = g_Pz_P[:,2]
            # Perform the second derivatives of k and P
            k_xx = tape.gradient(g_Ux_Ux, dummy)
            k_yy = tape.gradient(g_Uy_Uy, dummy)
            k_zz = tape.gradient(g_Uz_Uz, dummy)
            P_xx = tape.gradient(g_Px_Px, dummy_P)
            P_yy = tape.gradient(g_Py_Py, dummy_P)
            P_zz = tape.gradient(g_Pz_Pz, dummy_P)

            # Reshape pressures to match u shape
            P = tf.reshape(tf.stack(q*[P], axis=1), [P.shape[0], q])
            P_x = tf.reshape(tf.stack(q*[P_x], axis=1), [P_x.shape[0], q])
            P_y = tf.reshape(tf.stack(q*[P_y], axis=1), [P_y.shape[0], q])
            P_z = tf.reshape(tf.stack(q*[P_z], axis=1), [P_z.shape[0], q])
            P_xx = tf.reshape(tf.stack(q*[P_xx], axis=1), [P_xx.shape[0], q])
            P_yy = tf.reshape(tf.stack(q*[P_yy], axis=1), [P_yy.shape[0], q])
            P_zz = tf.reshape(tf.stack(q*[P_zz], axis=1), [P_zz.shape[0], q])

            # Rename for PDE
            u = k
            u_x = k_x
            u_y = k_y
            u_z = k_z
            u_xx = k_xx
            u_yy = k_yy
            u_zz = k_zz

            # Scale derivatives from input scaling
            x_scale = 0.5*(input_max[0] - input_min[0])
            y_scale = 0.5*(input_max[1] - input_min[1])
            z_scale = 0.5*(input_max[2] - input_min[2])
            P_x = P_x/x_scale
            P_y = P_y/y_scale
            P_z = P_z/z_scale
            P_xx = P_xx/x_scale**2
            P_yy = P_yy/y_scale**2
            P_zz = P_zz/z_scale**2
            u_x = u_x/x_scale
            u_y = u_y/y_scale
            u_z = u_z/z_scale
            u_xx = u_xx/x_scale**2
            u_yy = u_yy/y_scale**2
            u_zz = u_zz/z_scale**2

            # Define PDE
            u_t = (mu/output_max)*(1-P)*C_p + u*(eta_x*P_x + eta_y*P_y + eta_z*P_z) + eta*(u_x*P_x + u_y*P_y + u_z*P_z) + eta*u*(P_xx + P_yy + P_zz) - k_d_nd*u

            # Apply IRK Scheme to solve for predicted initial and final snapshots
            U0_pred = u - dt*tf.matmul(u_t, tf.transpose(IRK_alpha))
            U1_pred = u - dt*tf.matmul(u_t, tf.transpose(IRK_alpha - IRK_beta))

            # Calculate loss from comparing predicted and exact snapshots
            MSE = tf.reduce_mean(tf.square(U0 - U0_pred)) + tf.reduce_mean(tf.square(U1 - U1_pred)) + 0.01*tf.reduce_mean(tf.square((1-tumour_area)*P))

            # Return the loss
            return MSE

        def call(self, pred):
            self.add_loss(self.custom_loss(pred))
            loss_value = self.custom_loss(pred)
            return loss_value # output of loss layer is the loss value

    # Add Loss_Layer to model
    My_Loss = Loss_Layer()(Prediction)

    # Create trainable model with custom loss function
    model_loss = tf.keras.models.Model(inputs=Inputs, outputs=My_Loss)

if train_model == 'on':

    # Define a custom callback to print and record
    class CustomCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, loss, logs=None):
            loss = loss['loss']
            print('Epoch:', epoch+1, '... Loss:', round(loss, 10))

    # Compile model for Adam optimization
    model_loss.compile(optimizer=tf.keras.optimizers.Adam())
    # Execute Adam optimization
    history = model_loss.fit(train_inputs, None, epochs=Adam_Epochs, batch_size=train_inputs.shape[0], verbose=0, callbacks=[CustomCallback()])
    # Get losses from Adam optimization
    loss_tracking = history.history['loss']
    # Set up for L-BFGS optimization
    func = function_factory(model_loss, tf.keras.losses.MeanSquaredError(), train_inputs, np.zeros(train_inputs.shape[0]))
    init_params = tf.dynamic_stitch(func.idx, model_loss.trainable_variables)
    # Perform L-BFGS optimization
    results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=func, initial_position=init_params, max_iterations=LBFGS_Epochs)
    # Add loss from L-BFGS to tracking
    for i in range(len(func.history)):
        loss_tracking.append(func.history[i].numpy())

    # Get elapsed time
    t1 = time.time()
    Total_Time = t1-t0

if printing == 'on':
    # Print results
    print('#######################################')
    print('Total Time Elapsed:', round(Total_Time/60, 2), 'minutes')
    print('#######################################')

if saving == 'on':
    # Predict using model
    k_and_P_pred = model_prediction.predict(train_inputs)
    # Create predicted solution
    k_pred = k_and_P_pred[:,:-1]
    k_pred = k_pred.reshape((data_shape[0], data_shape[1], data_shape[2], q))
    train_outputs_t0 = train_outputs_t0.reshape((data_shape[0], data_shape[1], data_shape[2], 1))
    train_outputs_t1 = train_outputs_t1.reshape((data_shape[0], data_shape[1], data_shape[2], 1))
    U_pred = np.concatenate((train_outputs_t0, k_pred, train_outputs_t1), axis=3)
    U_pred = output_max*U_pred

    # Make predicted pressure
    P_pred = k_and_P_pred[:,-1]
    P_pred = P_pred.reshape((data_shape[0], data_shape[1], data_shape[2]))
    P_pred = 0.5*(np.tanh(P_pred) + 1)
    P_pred = P_v*P_pred

    # Reload unscaled parameters for saving
    u = data['u'][0]
    tumour_area = data['tumour_area'][0]
    dt = data['dt'][0]
    xyz = data['x'][0]
    C_p = data['C_p'][0]
    C_p_params = data['C_p_params'][0]
    param_dict = {'P_v' : P_v, 'k_d' : k_d,
                  'L_pS_over_V_normal' :  L_pS_over_V_normal, 'L_pS_over_V_tumour' :  L_pS_over_V_tumour,
                  'sigma_normal' : sigma_normal, 'sigma_tumour' : sigma_tumour,
                  'f_normal' : f_normal, 'f_tumour' : f_tumour,
                  'K_normal' : K_normal, 'K_tumour' : K_tumour}

    # Save the arrays
    scipy.io.savemat(output_file, mdict={'U_pred' : U_pred, 'P_pred' : P_pred, 'U_measure' : u, 'tumour_area' : tumour_area,
                                                                                    'dt' : dt, 'Pred_times' : times_for_C_p,  'x' : xyz, 'C_p' : C_p, 'C_p_params' : C_p_params,
                                                                                    'param_dict' : param_dict, 'loss' : loss_tracking})














# end
