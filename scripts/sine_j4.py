#!/usr/bin/env python3
import math
import sys
import threading

import numpy as np
import rospy
from sensor_msgs.msg import JointState
from wam_haptic_dmps.udp_handler import TeleopUDPHandler


class SineJ4:
    def __init__(self, remote_ip="127.0.0.1", send_port=5556, DOF=7):
        # UDP connection (send only)
        self.udp = TeleopUDPHandler(remote_ip, send_port, recv_port=None)

        # State Management
        self.state = "IDLE"  # Possible: IDLE, ROLLOUT, QUIT
        self.running = True

        # Start Input Thread
        self.input_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self.input_thread.start()

        self.last_joints = None
        self.initial_joints = None  # To store joint positions at the start of rollout
        self.DOF = DOF
        self.time = 0.0

        # Subscriber to get joint states
        self.follower_position_subscriber = rospy.Subscriber(
            "/follower/joint_states", JointState, self.follower_pos_callback
        )

        print("--- SineJ4 Initialized ---")
        print("Commands: [r] Rollout, [i] Idle, [q] Quit")

    def _keyboard_loop(self):
        """
        Runs in background. Waits for user input without blocking the main loop.
        """
        while self.running and not rospy.is_shutdown():
            try:
                cmd = input().strip().lower()

                if cmd == 'r':
                    if self.last_joints is not None:
                        self.initial_joints = list(self.last_joints)  # Save initial joint positions
                        self.time = 0.0
                        self.state = "ROLLOUT"
                        print(f"--> Switched to: {self.state}")
                    else:
                        print("Cannot start rollout: No joint state received yet.")
                elif cmd == 'i':
                    self.state = "IDLE"
                    print(f"--> Switched to: {self.state}")
                elif cmd == 'q':
                    self.state = "QUIT"
                    self.running = False
                    rospy.signal_shutdown("User Quit")
                else:
                    print(f"Unknown command: {cmd}. Use 'r' for rollout, 'i' for idle, or 'q' to quit.")

            except EOFError:
                break  # Handle Ctrl+D

    def follower_pos_callback(self, msg: JointState):
        """
        Callback for receiving WAM joint states. Updates self.last_joints.
        """
        self.last_joints = msg.position[:self.DOF]

    def run(self):
        """
        Main Control Loop (Real-time, e.g. 50Hz)
        """
        dt = 0.02  # 50Hz for updates
        rate = rospy.Rate(1 / dt)

        self.time = 0.0
        max_angle = 2.0  # Maximum allowable angle in radians for safety
        min_angle = 0.5

        while not rospy.is_shutdown() and self.running:
            if self.state == "ROLLOUT":
                if self.initial_joints is not None:
                    # Apply sine wave motion to joint 4 with clamping
                    target_positions = list(self.initial_joints)  # Use saved initial joint positions
                    sine_wave = 0.5 * math.sin(2 * math.pi * 0.01 * self.time)  # 0.5 rad amplitude, 0.5 Hz frequency
                    target_positions[3] += sine_wave  
                    target_positions[3] = np.clip(target_positions[3], min_angle, max_angle)
                    # rospy.loginfo(f"{target_positions[3]} - {self.last_joints[3]} - {sine_wave}")
                    rospy.loginfo(f"{target_positions[3] - self.initial_joints[3]} - {sine_wave}")
                    self.udp.send_data(target_positions, [0.0] * self.DOF, [0.0] * self.DOF)
                    self.time += dt

            elif self.state == "IDLE":
                pass  # Do nothing

            rate.sleep()

        self.udp.close()

if __name__ == "__main__":
    rospy.init_node('sine_j4')

    # Initialize
    sine_j4 = SineJ4(remote_ip="127.0.0.1", send_port=5556)

    # Start the main loop
    try:
        sine_j4.run()
    except rospy.ROSInterruptException:
        pass
