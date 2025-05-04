import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
dt = 1.0  # time step
num_steps = 50

# True initial state [x, v]
true_x = np.array([0, 1])  # starting at x=0, velocity=1 m/s

# Matrices for Kalman Filter
F = np.array([[1, dt], [0, 1]])  # state transition model
H = np.array([[1, 0]])           # measurement model (only position)

Q = np.array([[0.01, 0], [0, 0.01]])  # process noise covariance
R = np.array([[0.5]])                 # measurement noise covariance

# Initial Kalman Filter guesses
x_est = np.array([0, 0])   # initial guess: at 0 position, 0 velocity
P_est = np.eye(2)          # initial covariance

# Store results
true_positions = []
measured_positions = []
estimated_positions = []

# Simulation loop
for step in range(num_steps):
    # Simulate true motion
    true_x = F @ true_x + np.random.multivariate_normal([0, 0], Q)

    # Simulate noisy GPS measurement
    z = H @ true_x + np.random.normal(0, np.sqrt(R[0,0]))

    # === Kalman Filter Prediction ===
    x_pred = F @ x_est
    P_pred = F @ P_est @ F.T + Q

    # === Kalman Filter Update ===
    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
    x_est = x_pred + K.flatten() * (z - H @ x_pred)
    P_est = (np.eye(2) - K @ H) @ P_pred

    # Save data
    true_positions.append(true_x[0])
    measured_positions.append(z[0])
    estimated_positions.append(x_est[0])

print(measured_positions)
# === Plot results ===
plt.figure(figsize=(10,6))
plt.plot(true_positions, label='True Position')
plt.plot(measured_positions, 'o', label='Measured Position (Noisy)', markersize=3)
plt.plot(estimated_positions, label='Estimated Position (Kalman Filter)')
plt.xlabel('Time Step')
plt.ylabel('Position')
plt.legend()
plt.title('Kalman Filter for 1D Straight-Line Robot Motion')
plt.grid()
plt.show()
