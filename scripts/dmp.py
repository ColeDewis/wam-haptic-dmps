#!/usr/bin/env python3
import sys
import threading

import rospy
from movement_primitives.dmp import DMP
from sensor_msgs.msg import JointState
from wam_haptic_dmps.udp_handler import TeleopUDPHandler
import numpy as np


class DMPLearner:
    def __init__(self, remote_ip="127.0.0.1", send_port=5556, DOF=7, dt=0.02):
        # UDP connection (send only)
        self.udp = TeleopUDPHandler(remote_ip, send_port, recv_port=None, DOF=DOF)
        
        # 2. State Management
        self.state = "IDLE"  # Possible: IDLE, LEARN, ROLLOUT, QUIT
        self.running = True
        self.learning_active = False  # Tracks if learning is active

        # 3. Start Input Thread
        # daemon=True means this thread dies automatically when the main program quits
        self.input_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self.input_thread.start()
        
        # TODO: switch to learder follower setup
        self.follower_position_subscriber = rospy.Subscriber("/follower/joint_states", JointState, self.follower_pos_callback)

        self.trajectory_buffer = []
        self.last_joints = None
        self.dof = DOF
        self.dmp = DMP(n_dims=DOF, dt=dt)
        self.dmp_goal = None
        self.dt = dt
        self.last_record_time = rospy.Time(0)

        print("--- DMP Learner Initialized ---")
        print("Commands: [s] Start Learning, [r] Rollout, [i] Idle, [q] Quit")

    def _keyboard_loop(self):
        """
        Runs in background. Waits for user input without blocking the main loop.
        """
        while self.running and not rospy.is_shutdown():
            try:
                cmd = input().strip().lower()
                
                if self.state == "LEARN" and self.learning_active:
                    # Handle commands during learning
                    if cmd == 'save':
                        self._save_trajectory()
                        self.state = "IDLE"
                        self.learning_active = False
                        print("--> Trajectory saved. Returning to IDLE.")
                    elif cmd == 'restart':
                        self.trajectory_buffer = []
                        print("--> Trajectory buffer cleared. Restarting learning.")
                    elif cmd == 'quit':
                        self.state = "IDLE"
                        self.learning_active = False
                        print("--> Learning aborted. Returning to IDLE.")
                    else:
                        print(f"Unknown command: {cmd}. Options: [save], [restart], [quit]")
                else:
                    # Handle commands outside of learning
                    if cmd == 's':
                        self.state = "LEARN"
                        self.learning_active = True
                        self.trajectory_buffer = []  # Clear buffer for new learning session
                        print(f"--> Switched to: {self.state}. Options: [save], [restart], [quit]")
                    elif cmd == 'r':
                        self.state = "ROLLOUT"
                        print(f"--> Switched to: {self.state}")
                    elif cmd == 'i':
                        self.state = "IDLE"
                        print(f"--> Switched to: {self.state}")
                    elif cmd == 'q':
                        self.state = "QUIT"
                        self.running = False
                        rospy.signal_shutdown("User Quit")
                    else:
                        print(f"Unknown command: {cmd}")
                    
            except EOFError:
                break  # Handle Ctrl+D

    def _save_trajectory(self):
        """
        Save the recorded trajectory and train the DMP.
        """
        if self.trajectory_buffer:
            print("Training DMP with recorded trajectory...")
            n_steps = len(self.trajectory_buffer)
            execution_time = (n_steps - 1) * self.dt
            T = np.linspace(0, execution_time, n_steps)
            self.dmp.imitate(T, np.array(self.trajectory_buffer))
            # we need to specify how long it takes or it will execute trajectory in 1 second by default
            self.dmp.set_execution_time_(execution_time)
            
            print("DMP training complete.")
            self.trajectory_buffer = []  # Clear the buffer after saving
        else:
            print("No trajectory data to save.")

    def follower_pos_callback(self, msg: JointState):
        """
        Callback for receiving WAM joint states. This is where you would
        typically record data during the LEARN phase or use it during ROLLOUT.
        """
        self.last_joints = msg.position[:self.dof] 
        self.last_jv = msg.velocity[:self.dof]
        if self.state == "LEARN" and self.learning_active:
            # Record the data for learning
            now = rospy.Time.now()
            
            # we want incoming traj to match what we send in terms of timesteps
            if (now - self.last_record_time).to_sec() >= self.dt:
                self.trajectory_buffer.append(msg.position[:self.dof])
                self.last_record_time = now

    # def external_torque_callback(self, msg: GravityTorque):
    #     # TODO not sure we actually need this here at all for now tbh
    #     pass

    def run(self):
        """
        Main Control Loop (Real-time, e.g. 500Hz)
        """
        rate = rospy.Rate(1/self.dt)
        count = 0
        
        while not rospy.is_shutdown() and self.running:
            
            # --- STATE MACHINE ---
            if self.state == "IDLE":
                pass
                
            elif self.state == "LEARN":
                # Learning is handled in the callback and keyboard loop
                pass 
            
            elif self.state == "ROLLOUT":
                # Execute the learned DMP
                if self.last_joints is not None and not (self.dmp.start_y == self.dmp.goal_y).all():
                    y, yd = self.dmp.step(self.last_joints, self.last_jv)
                    if (yd < 0.0001).all():
                        self.state = "IDLE"
                        print(f"reached end of DMP trajectory after {count} steps, entering IDLE.")
                    else:
                        print(count, y, yd)
                        count += 1
                        self.udp.send_data(y, np.zeros(self.dof), np.zeros(self.dof))
                else:
                    print("likely not trained a valid DMP yet")

            rate.sleep()

        self.udp.close()

if __name__ == "__main__":
    rospy.init_node('dmp_learner')
    
    # Initialize
    learner = DMPLearner(remote_ip="127.0.0.1", send_port=5556, DOF=7)
    
    # Start the main loop
    try:
        learner.run()
    except rospy.ROSInterruptException:
        pass
