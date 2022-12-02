import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

#simulated sensor/measurements producing function
def get_measurements():
    t = np.linspace(0,25,num=25) #because 1 per sec (1 hertz)
    num_measurements = len(t)
    #initialize pos
    x = 2900
    y = 2900
    #velocity is 22 m/s
    vel = 22.0
    #initialize empty arrays for output later
    t_time = []
    t_x = []
    t_y = []
    t_r = []
    t_theta = []
    #get real position data : simulation using cartesian to get new positions and converting that to polar
    for i in range(0, num_measurements):
        dT = 1.0
        #store time of pos update
        t_time.append(t[i])
        #compute the new cartesian coords
        x = x+vel*dT
        y = y
        #store
        t_x.append(x)
        t_y.append(y)
        #get r and theta and store them too
        r = np.sqrt(x*x + y*y)
        t_r.append(r)
        theta = np.arctan2(x,y) * 180/np.pi
        t_theta.append(theta)
    #initialize lists for storing polar meas
    m_r = []
    m_theta = []
    m_cov = []
    #theta standard deviation = 9 milliradians converted to degrees
    sig_theta = 0.009*180/np.pi
    sig_r = 30 #range standard deviation = 30 meters
    m_x = []
    m_y = []
    for ii in range(0,len(t_time)):
        #compute error for each measurement by taking the max between 0.25 and the standard deviation and th randomly generated normal error
        #this makes sure error exists
        temp_sig_theta = np.maximum(sig_theta * np.random.randn(), 0.25*sig_theta)
        temp_sig_r = np.maximum(sig_r*np.random.randn(), 0.25*sig_r)
        #save'em
        temp_theta = t_theta[ii] + temp_sig_theta
        temp_r = t_r[ii] + temp_sig_r
        #save off the meas data
        m_theta.append(temp_theta)
        m_r.append(temp_r)
        m_cov.append(np.array([[temp_sig_r*temp_sig_r,0], [0,temp_sig_theta*temp_sig_theta]]))
        m_x.append(temp_r*np.sin(temp_theta*np.pi/180))
        m_y.append(temp_r*np.cos(temp_theta*np.pi/180))
    return [m_r,m_theta,m_cov,t_r,t_theta,t_time,t_x,t_y,m_x,m_y]

        
#function defining Extended Kalman Filter Algorithmn Implementation
# this example is of polar position measurements (r, theta) and a pos + vel state (x,y,xdot,ydot) [UNITS IN m & m/s]
# what makes this an EKF is the nonlinearity in the state and measurement relationship (r = x^2 + y^2, & theta = tan^-1(y/x))
# Normal Kalman Filters cannot be implemented successfully on nonlinearities in the system model (in this case the H (state to measurement matrix))
def ekf(z, update_number): # z is measurement
    dt = 1.0 #update frequency of 1 second for convenience
    no = update_number #convenience
    if update_number == 0: #initialization
        #compute pos values from polar measurement z = [r, theta]
        x_temp = z[0][no]*np.sin(z[1][no]*np.pi/180)
        y_temp = z[0][no]*np.cos(z[1][no]*np.pi/180)
        #state
        #if confused about matrix labelling please refer to the comments on the initialization step of the KF_example.py script
        ekf.x = np.array([[x_temp],[y_temp],[0],[0]])
        ekf.P = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
        ekf.A = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
        ekf.R = z[2][no]
        ekf.Q = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
        #initialize empty residual and Kalman Gain (K) such that stuff outputs
        residual = np.array([[0,0],[0,0]])
        K = np.array([[0,0],[0,0],[0,0],[0,0]])
    if update_number == 1: #reinitialize!
        x_0 = ekf.x[0][0]
        y_0 = ekf.x[1][0]
        x_temp = z[0][no]*np.sin(z[1][no]*np.pi/180)
        y_temp = z[0][no]*np.cos(z[1][no]*np.pi/180)
        #calculate velocity this time 'round! (it's in m/s)
        vx_temp = (x_temp - x_0)/dt
        vy_temp = (y_temp - y_0)/dt
        #set params
        ekf.x = np.array([[x_temp],[y_temp],[vx_temp],[vy_temp]])
        ekf.P = np.array([[100, 0, 0, 0], [0,100,0,0],[0,0,250,0],[0,0,0,250]]) #initialized to random large values on diagonal
        ekf.A = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
        ekf.R = z[2][no]
        ekf.Q = np.array([[20,0,0,0],[0,20,0,0],[0,0,4,0],[0,0,0,4]]) #4.5 std dev in pos and 2 m/s std dev in vel
        residual = np.array([[0,0],[0,0]])
        K = np.array([[0,0],[0,0],[0,0],[0,0]])
    if update_number > 1: #rest of filter algo past init and reinit steps!
        #prediction
        x_p = ekf.A.dot(ekf.x)
        P_p = ekf.A.dot(ekf.P).dot(ekf.A.T) + ekf.Q
        x_1 = x_p[0][0]
        y_1 = x_p[1][0]
        denominator_1 = np.sqrt(x_1*x_1+y_1*y_1)
        ekf.H = np.array([[x_1/denominator_1, y_1/denominator_1, 0, 0], [y_1/denominator_1, -x_1/denominator_1, 0, 0]])
        ekf.H_t = np.array([[x_1/denominator_1,y_1/denominator_1], [y_1/denominator_1, -x_1/denominator_1],[0,0],[0,0]]) #demo purposes can use numpy's transpose
        ekf.R = z[2][no]
            #compute Kalman Gain (K)
        S = ekf.H.dot(P_p).dot(ekf.H_t) + ekf.R
        K = P_p.dot(ekf.H_t).dot(np.linalg.inv(S))
        #estimation of state
        z_temp = np.array([[z[0][no]],[z[1][no]]])
            #convert state prediction to polar
        x_prediction = x_p[0][0]
        y_prediction = x_p[1][0]
        sum_squares = x_prediction*x_prediction + y_prediction*y_prediction
        r_prediction = np.sqrt(sum_squares)
        theta_prediction = np.arctan2(x_prediction,y_prediction) * 180/np.pi
        mini_h = np.array([[r_prediction],[theta_prediction]])
        #compute dif of state and meas for data time
        residual = z_temp - mini_h
        #calculate NEW ESTIMATE       
        ekf.x = x_p + K.dot(residual)
        ekf.P = P_p - K.dot(ekf.H).dot(P_p)
    return [ekf.x[0], ekf.x[1], ekf.P, ekf.x[2], ekf.x[3], K, residual];

#TEST AND PLOTTING CODE BELOW
#TEST CODE 
f_x = []
f_y = []
f_x_sig = []
f_y_sig = []
f_xv = []
f_yv = []
f_xv_sig = []
f_yv_sig = []

z = get_measurements()       
for iii in range(0,len(z[0])):
    f = ekf(z,iii)
    f_x.append(f[0])
    f_y.append(f[1])
    f_xv.append(f[3])
    f_yv.append(f[4])
    f_x_sig.append(np.sqrt(f[2][0][0])) 
    f_y_sig.append(np.sqrt(f[2][1][1]))

plot1 = plt.figure(1)
plt.grid(True)
plt.plot(z[5], z[3])
plt.scatter(z[5],z[0])
plt.title('Actual Range vs Measured Range')
plt.legend(['Ship Actual Range', 'Ship Measured Range'])
plt.ylabel('Range (meters)')
plt.xlabel('Update Number')

plot2 = plt.figure(2)
plt.grid(True)
plt.plot(z[5],z[4])
plt.scatter(z[5],z[1])
plt.title('Actual Azimuth vs Measured Azimuth')
plt.legend(['Vehicle Actual Azimuth', 'Ship Measured Azimuth'])
plt.ylabel('Azimuth (degrees)')
plt.xlabel('Update Number')

plot3 = plt.figure(3), plt.grid(True)
plt.plot(z[5],f_xv)
plt.plot(z[5],f_yv)
plt.title('Velocity Estimate On Each Measurement Update \n', fontweight = "bold")
plt.legend(['X Velocity Estimate', 'Y Velocity Estimate'])

#get error in r
e_x_err = []
e_x_3sig = []
e_x_3sig_neg = []
e_y_err = []
e_y_3sig = []
e_y_3sig_neg = []
for m in range(0,len(z[0])):
    e_x_err.append(f_x[m]-z[6][m])
    e_x_3sig.append(3*f_x_sig[m])
    e_x_3sig_neg.append(-3*f_x_sig[m])
    e_y_err.append(f_y[m]-z[7][m])
    e_y_3sig.append(3*f_y_sig[m])
    e_y_3sig_neg.append(-3*f_y_sig[m])

plot4 = plt.figure(4), plt.grid(True)
line1 = plt.scatter(z[5],e_x_err)
line2, = plt.plot(z[5], e_x_3sig, color='green')
plt.plot(z[5],e_x_3sig_neg, color= 'green')
plt.ylabel('Position Error (meters')
plt.xlabel('Update Number')
plt.title('X Position Estimate Error Containment \n', fontweight="bold")
plt.legend([line1,line2,], ['X Position Error', '3 Sigma Error Bound'])

plot5 = plt.figure(5), plt.grid(True)
yline1 = plt.scatter(z[5], e_y_err)
yline2, = plt.plot(z[5], e_y_3sig, color='green')
plt.plot(z[5],e_y_3sig_neg, color='green')
plt.ylabel('Position Error (meters)')
plt.xlabel('Update Number')
plt.title('Y Position Estimate Error Containment \n',fontweight="bold")
plt.legend([yline1,yline2,],['Y Position Error', '3 Sigma Error Bound'])
plt.show()

