
Kalman_Filter_Example ---------------------------------

Simple KF example with simplified dynamics and noise + error, good for reference, teaching, and as a template.
Models for Kalman Filter and Extended Kalman Filter

Proof these examples demonstrate working KF algorithm implementations:

Kalman Filter for Position and Velocity Tracking

![Figure_2](https://user-images.githubusercontent.com/95187192/205463058-7515e79e-a573-498d-bf4a-5a45e7d213b4.png)

^Actual velocity is 60 m/s but initial estimate is set to 20. Clearly, it rises towars and sustains an estimate velocity near 60 m/s after only a few updates.

![Figure_4](https://user-images.githubusercontent.com/95187192/205463175-0f9304d9-1572-4e5f-a3c8-813271ac4791.png)

^Kalman Gains move towards a steady state as estimation process starts to converge, Velocity is only a little in favor of the estimate since it is 0.5 and close to an even weight of estimate and measurement whereas the position is significantly more favoring of the estimate due to the lower gain.

![Figure_3](https://user-images.githubusercontent.com/95187192/205463334-f996860d-1aa9-41c1-a32b-d33ee660b691.png)

^Shows general filtering of measurement position error noise.

Extended Kalman Figure Example ----------------------------

![Figure_3](https://user-images.githubusercontent.com/95187192/205463564-954608de-99cb-475b-a024-edce06685c8c.png)

^Actual velocity in x direction was set to be around 20 m/s with some added noise. Initial estimate was 0 m/s so it does approach that. What makes this an EKF example is that the state to measurement matrix is not linear. It is jacobian of polar to cartesian conversion. This would be a good example of a radar (polar sensor) being a the sensor that would update the filter. (y direction actual velocity set to around zero.

![Figure_4](https://user-images.githubusercontent.com/95187192/205463689-27aa4f6a-3134-4c15-b5d7-b42bb71b7cbf.png)

![Figure_5](https://user-images.githubusercontent.com/95187192/205463695-eebf1b7f-72a3-45a3-a957-ef61b05cecad.png)

^these show that position error is mostly contained within 3-sigma (standard devations) for this example. Didn't spend time tuning Q matrix etc since this is just an example. However this code should be relatively easy to digest.

Credit again to  Franklin's Book: A Kalman Filter Made Easy

