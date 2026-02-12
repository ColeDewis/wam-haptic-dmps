#!/usr/bin/env python3
import sys
import threading

import rospy
from movement_primitives.dmp import DMP
from sensor_msgs.msg import JointState
from wam_haptic_dmps.udp_handler import TeleopUDPHandler
from wam_teleop.msg import GravityTorque


class DMPLearner:
    def __init__(self, remote_ip="127.0.0.1", send_port=5556, DOF=7):
        # UDP connection (send only)
        self.udp = TeleopUDPHandler(remote_ip, send_port, recv_port=None)
        
        # 2. State Management
        self.state = "IDLE"  # Possible: IDLE, LEARN, ROLLOUT, QUIT
        self.running = True
        self.learning_active = False  # Tracks if learning is active

        # 3. Start Input Thread
        # daemon=True means this thread dies automatically when the main program quits
        self.input_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self.input_thread.start()
        
        self.follower_position_subscriber = rospy.Subscriber("/follower/joint_states", JointState, self.follower_pos_callback)
        self.leader_external_torque_subscriber = rospy.Subscriber("/leader/external_torque", GravityTorque, self.external_torque_callback)

        self.trajectory_buffer = []
        self.last_joints = None
        self.DOF = DOF
        self.dmp = DMP(n_dims=DOF)
        self.dmp_goal = None

        print("--- DMP Learner Initialized ---")
        print("Commands: [s] Start Learning, [f] Finetune, [r] Rollout, [i] Idle, [q] Quit")

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
                    elif cmd == 'f':
                        self.state = "FINETUNE"
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
            self.dmp.imitate(self.trajectory_buffer)
            
            print("DMP training complete.")
            self.trajectory_buffer = []  # Clear the buffer after saving
        else:
            print("No trajectory data to save.")

    def follower_pos_callback(self, msg: JointState):
        """
        Callback for receiving WAM joint states. This is where you would
        typically record data during the LEARN phase or use it during ROLLOUT.
        """
        self.last_joints = msg.position[:self.DOF] 
        self.last_jv = msg.velocity[:self.DOF]
        if self.state == "LEARN" and self.learning_active:
            # Record the data for learning
            self.trajectory_buffer.append(msg.position[:self.DOF]) 

    def external_torque_callback(self, msg: GravityTorque):
        # TODO not sure we actually need this here at all for now tbh
        pass

    def run(self):
        """
        Main Control Loop (Real-time, e.g. 500Hz)
        """
        dt = 0.02  # 50Hz for DMP updates, can be adjusted as needed
        
        rate = rospy.Rate(1/dt)
        
        while not rospy.is_shutdown() and self.running:
            
            # --- STATE MACHINE ---
            if self.state == "IDLE":
                pass
                
            elif self.state == "LEARN":
                # Learning is handled in the callback and keyboard loop
                pass 
            
            elif self.state == "FINETUNE":
                # have to send data as in rollout but also then
                # read back the new positions
                pass
            
            elif self.state == "ROLLOUT":
                # Execute the learned DMP
                # self.dmp.configure()
                if self.last_joints is not None:
                    y, yd = self.dmp.step(self.last_joints, self.last_jv)
                    self.udp.send_data(y, yd, [0.0]*self.DOF)

            rate.sleep()

        self.udp.close()

if __name__ == "__main__":
    rospy.init_node('dmp_learner')
    
    # Initialize
    learner = DMPLearner(remote_ip="127.0.0.1", send_port=5556)
    
    # Start the main loop
    try:
        learner.run()
    except rospy.ROSInterruptException:
        pass