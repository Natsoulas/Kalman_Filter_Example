# Kalman Filter Template script given in Position and Velocity Tracking Example
#       this pos and vel tracking application assumes constant vel (no acceleration)
# based on Franklin's KF book
# Import Libraries for LinAlg & Plots
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

def get_Measurement(update_number):
    # initialize the function variables to be true values
    if update_number == 1:
        get_Measurement.currentPosition = 0 # unit: meters
        get_Measurement.currentVelocity  = 60 # unit: seconds
    
    # define dt between each update
    dt = 0.1 #unit: seconds
    
    #compute random noise for position & Velocity
    pos_noise = 8 * np.random.randn(1)
    vel_noise = 8 * np.random.randn(1)
    
    # z is or the position measurement is defined by simple EOM 
    
    z = get_Measurement.currentPosition + get_Measurement.currentVelocity*dt + pos_noise
    
    #reset true position to be the current pos minus the noise added in previous step
    get_Measurement.currentPosition = z - pos_noise
    #update vel to include noise (which equates to dv's or accelerations which is realistic noise in this case)
    get_Measurement.currentVelocity = 60 + vel_noise
    return [z, get_Measurement.currentPosition, get_Measurement.currentVelocity]

#filter function to actually implement system model and filtering
def kalman_filter(z, update_number):
    dt = 0.1
    # INITIALIZATION STEP:
    #sys model, state estimate & covariance
    if update_number == 1:
        # x is the state estimate
        kalman_filter.x = np.array([[0],
                                    [20]])
        # P is the state covariance matrix
        kalman_filter.P = np.array([[5,0],
                                    [0,5]])
        # A is the state transition matrix
        kalman_filter.A = np.array([[1, dt],
                                    [0,1]])
        # H is the state to measureent matrix
        kalman_filter.H = np.array([[1, 0]])
        # H_t is H transpose (notation muhim!)
        kalman_filter.H_t = np.transpose(kalman_filter.H)
        # R is the measurement covariance (error) matrix
        kalman_filter.R = 10
        # Q is the process noise covariance matrix
        kalman_filter.Q = np.array([[1,0],
                                    [0,3]])
    
    # PREDICTION STEP (SUBSCRIPT p)
    x_p = kalman_filter.A.dot(kalman_filter.x)
    P_p = kalman_filter.A.dot(kalman_filter.P).dot(kalman_filter.A.T) + kalman_filter.Q
    
    # KALMAN GAIN COMPUTATION STEP
     # S is innovation matrix (what's inversed (don't read into the name too much))
    S = kalman_filter.H.dot(P_p).dot(kalman_filter.H_t) + kalman_filter.R 
    K = P_p.dot(kalman_filter.H_t)*(1/S)
    
    # ESTIMATION STEP
    residual = z - kalman_filter.H.dot(x_p)
    kalman_filter.x = x_p + K*residual
    kalman_filter.P = P_p - K.dot(kalman_filter.H).dot(P_p)
    
    return [kalman_filter.x[0],kalman_filter.x[1], kalman_filter.P, K]

def testKF():
    #define timespan of test (seconds ofc)
    dt = 0.1
    t = np.linspace(0, 10, num = 300)
    num_measurements = len(t)
    
    # initialize arrays for plotting data collection
    meas_time = []
    meas_pos = []
    meas_dif_pos = []
    est_dif_pos = []
    est_pos = []
    est_vel = []
    pos_bound_3sig = []
    pos_gain = []
    vel_gain = []
    
    #loop through measurements
    for k in range(1, num_measurements):
        #create most recent measurement
        z = get_Measurement(k)
        #call kalman filter to get new state f
        f = kalman_filter(z[0],k)
        #save off plotting data
        meas_time.append(k)
        meas_pos.append(z[0])
        meas_dif_pos.append(z[0]-z[1])
        est_dif_pos.append(f[0]-z[1])
        est_pos.append(f[0])
        est_vel.append(f[1])
        posVar = f[2]
        pos_bound_3sig.append(3*np.sqrt(posVar[0][0]))
        K = f[3]
        pos_gain.append(K[0][0])
        vel_gain.append(K[1][0])
    return [meas_time, meas_pos, est_pos, est_vel, meas_dif_pos, est_dif_pos, pos_bound_3sig, pos_gain, vel_gain]

# Kalman Filter Test Call
t = testKF()
# Plot Test Result
plot1 = plt.figure(1)
plt.scatter(t[0],t[1])
plt.plot(t[0],t[2])
plt.ylabel('Position')
plt.xlabel('Time')
plt.grid(True)

plot2 = plt.figure(2)
plt.plot(t[0],t[3])
plt.ylabel('Velocity (m/s)')
plt.xlabel('Update Number')
plt.title('Velocity Estimate For Each Measurement Update \n', fontweight = "bold")
plt.legend(['Estimate'])
plt.grid(True) 

plot3 = plt.figure(3)
plt.scatter(t[0],t[4], color = 'red')
plt.plot(t[0],t[5])
plt.legend(['Estimate','Measurement'])
plt.title('Position Errors On Each Measurement Update \n', fontweight="bold")
plt.ylabel('Position Error (meters)')
plt.xlabel('Update Number')
plt.grid(True)
plt.xlim([0,300])

plot4 = plt.figure(4)
plt.plot(t[0],t[7])
plt.plot(t[0],t[8])
plt.ylabel('Kalman Gain')
plt.xlabel('Update Number')
plt.grid(True)
plt.xlim([0,100])
plt.legend(['Position KG', 'Velocity KG'])
plt.title('Position and Velocity Gains \n', fontweight = "bold")

plt.show()