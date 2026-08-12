#!/usr/bin/env python3

import numpy as np
import rospy
import os
import signal
import struct
import time

from pynput import keyboard
from termcolor import cprint
from movement_primitives.dmp import DMP
from wam_haptic_dmps.udp_sender import UDPSender
from wam_haptic_dmps.udp_receiver import UDPReceiver
from wam_haptic_dmps.recorder import Recorder
from wam_haptic_dmps.precise_sleep import precise_wait

class DMPPlayer:
    def __init__(self, episode_idx=0, remote_ip="127.0.0.1", leader_send_port=10000, follower_send_port=20000, recv_port=6554, DOF=7, hz=10):
        self.horizon = 8

        self.udp_receiver = UDPReceiver(
            remote_ip, recv_port, DOF
        )
        self.leader_udp_sender = UDPSender(
            remote_ip, leader_send_port, DOF=DOF, horizon=self.horizon
        )
        self.follower_udp_sender = UDPSender(
            remote_ip, follower_send_port, DOF=DOF, horizon=self.horizon
        )

        self.send_interval = 0.4  # send interpolated points for 3 seconds
        self.last_send_time = 0.0

        self.inference_time = 1.0          # estimate of how long inf takes
        self.inference_time_alpha = 0.2    # EMA smoothing factor for the estimate

        self.loop_state = "IDLE"
        self.trajectory_buffer = []

        self.dof = DOF
        self.dt = 1/hz
        self.dmp = DMP(n_dims=DOF+1, dt=self.dt, n_weights_per_dim=20)
        self.dmp_goal = None

        self.recorder = Recorder(save_dir="/home/user/wam_ros/wam_ws/src/wam_haptic_dmps/dataset")
        self.trajectory_buffer = self.recorder.load_episode(episode_idx)

        print("Training DMP with recorded trajectory...")
        n_steps = len(self.trajectory_buffer)
        execution_time = (n_steps - 1) * self.dt
        T = np.linspace(0, execution_time, n_steps)
        self.dmp.imitate(T, np.array(self.trajectory_buffer))
        # we need to specify how long it takes or it will execute trajectory in 1 second by default
        self.dmp.set_execution_time_(execution_time)
        self.dmp_goal = np.array(self.trajectory_buffer[-1])
        self.dmp_output = None
        self.dmp_start_idx = 0
        
        print("DMP training complete.")

        # Set up Joystick
        self.joy_fd = None
        self._init_joystick()

        self.kb_listener = keyboard.Listener(on_press=self._on_key_press)
        self.kb_listener.start()
        cprint("Initialization complete. Running teleop loop...", "green")
        cprint("Controls: [R] Start/Stop", "cyan")

    def _init_joystick(self):
        """Initializes the joystick file descriptor in non-blocking mode."""
        try:
            self.joy_fd = os.open("/dev/input/js0", os.O_RDONLY | os.O_NONBLOCK)
            cprint("[SYSTEM] Successfully opened joystick /dev/input/js0", "green")
        except OSError:
            cprint(
                "[SYSTEM] Could not open joystick /dev/input/js0. Continuing without joystick.",
                "yellow",
            )

    def _poll_joystick(self):
        """Reads non-blocking events from the joystick."""
        if self.joy_fd is None:
            return

        try:
            while True:
                event_data = os.read(self.joy_fd, 8)
                if not event_data:
                    break

                time_msec, value, ev_type, number = struct.unpack("IhBB", event_data)

                # Remove the init event flag (0x80)
                ev_type &= ~0x80

                # ev_type == 1 means button event, value == 1 means pressed (down)
                if ev_type == 0x01 and value == 1:
                    # BLUETOOTHCTL connection
                    if number == 1:  # 'o' button
                        self._handle_start_action()
                    elif number == 0:  # 'x' button
                        print("press x")
                        self._handle_stop()
                    # SIXAXIS connection
                    if number == 13:  # 'o' button
                        print("press o")
                        self._handle_start_action()
                    elif number == 14:  # 'x' button
                        print("press x")
                        self._handle_stop()

        except BlockingIOError:
            pass
        except Exception as e:
            cprint(f"[SYSTEM] Error reading joystick: {e}", "red")

    def _on_key_press(self, key):
        """Asynchronous callback for keyboard events."""
        try:
            if hasattr(key, "char") and key.char is not None:
                k = key.char.lower()

                if k == "r":
                    self._handle_start_action()
                elif k == "d":
                    self._handle_stop()
        except Exception:
            pass

    def _handle_start_action(self):
        """Unified logic for 'r' / 's' on keyboard and 'o' on joystick."""
        if self.loop_state == "IDLE":
            self.loop_state = "ROLLOUT"
            self.dmp_output = None
            self.dmp_start_idx = 0
            self.last_send_time = 0.0
            cprint("\n[RECORDER] 🔴 EPISODE STARTED", "red", attrs=["bold"])

    def _handle_stop(self):
        """'d' on keyboard / 'x' on joystick: stop any action sending."""
        if self.loop_state == "ROLLOUT":
            self.loop_state = "IDLE"
            self.dmp_output = None
            self.dmp_start_idx = 0

            cprint("Press [R] or 'o' to start recording", "cyan")


    def shutdown(self):
        cprint("Cleaning up streams and windows...", "red")
        self.follower_udp_sender.close()
        self.leader_udp_sender.close()
        self.udp_receiver.close()
        if self.joy_fd is not None:
            try:
                os.close(self.joy_fd)
            except Exception:
                pass

        os._exit(0)
    
    def _read_state(self):
        state_dict = self.udp_receiver.receive_latest_data()
        status = state_dict is not None

        return status, state_dict

    def run(self):
        try:
            self.system_running = True

            def force_shutdown(signum, frame):
                cprint(
                    "\n[SYSTEM] Ctrl+C detected! Forcing main loop shutdown...", "red"
                )
                self.system_running = False

            signal.signal(signal.SIGINT, force_shutdown)

            t_start = time.monotonic()
            iter_idx = 0
            while self.system_running:
                t_cycle_end = t_start + (iter_idx + 1) * self.dt

                self._poll_joystick()

                # Read images
                state_status, obs = self._read_state()

                if not state_status:
                    cprint("waiting for messages", "yellow")
                else:
                    # Infer + send to WAM
                    if self.loop_state == "ROLLOUT":
                        if self.dmp_output is None:
                            # configure the DMP to start from current position, keeping same end goal.
                            self.dmp.configure(
                                start_y=np.array([*obs["follower_jp"], obs["gripper_pos"]]),
                                start_yd=np.array([*obs["follower_jv"], obs["gripper_vel"]]),
                                goal_y=self.dmp_goal,
                            )
                            _, self.dmp_output = self.dmp.open_loop()

                        time_to_chunk_end_s = obs["time_to_chunk_end_ns"] / 1e9
                        interval_elapsed = (time.time() - self.last_send_time) >= self.send_interval

                        should_infer = (time_to_chunk_end_s <= self.inference_time) and interval_elapsed

                        if should_infer:
                            t_infer_start = time.monotonic()

                            dmp_end_idx = min(self.dmp_start_idx + self.horizon, len(self.dmp_output))
                            action_chunk = self.dmp_output[self.dmp_start_idx:dmp_end_idx, :]

                            if action_chunk.shape[0] < self.horizon:
                                pad_count = self.horizon - action_chunk.shape[0]
                                pad = np.tile(action_chunk[-1], (pad_count, 1))
                                action_chunk = np.vstack([action_chunk, pad])

                            elapsed = time.monotonic() - t_infer_start
                            self.inference_time = (
                                self.inference_time_alpha * elapsed
                                + (1 - self.inference_time_alpha) * self.inference_time
                            )

                            print(self.dmp_start_idx)
                            print(action_chunk)
                            print(action_chunk.shape)
                            print(dmp_end_idx)

                            time_to_skip_ns = obs["time_to_chunk_end_ns"]
                            self.leader_udp_sender.send_action_chunk(action_chunk, time_to_skip_ns)
                            self.follower_udp_sender.send_action_chunk(action_chunk, time_to_skip_ns)
                            self.dmp_start_idx += self.horizon
                            self.last_send_time = time.time()

                            if dmp_end_idx == len(self.dmp_output):
                                print("reached end of trajectory")
                                self._handle_stop()

                precise_wait(t_cycle_end)
                iter_idx += 1


        except Exception as e:
            import traceback

            print("CRASHED WITH ERROR:")
            traceback.print_exc()

        finally:
            self.shutdown()

if __name__ == "__main__":
    rospy.init_node('dmp_player')
    
    # Initialize
    player = DMPPlayer(episode_idx=0, remote_ip="127.0.0.1", leader_send_port=10000, follower_send_port=20000, recv_port=6554, DOF=7)
    
    # Start the main loop
    try:
        player.run()
    except rospy.ROSInterruptException:
        pass
