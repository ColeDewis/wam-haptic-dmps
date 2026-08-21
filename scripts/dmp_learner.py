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

class DMPLearner:
    def __init__(self, remote_ip="127.0.0.1", recv_port=6554, DOF=7, hz=10):
        self.udp_receiver = UDPReceiver(
            remote_ip, recv_port, DOF
        )
        
        self.loop_state = "IDLE" 
        self.trajectory_buffer = []

        self.dof = DOF
        self.dt = 1/hz
        self.dmp = DMP(n_dims=DOF, dt=self.dt, n_weights_per_dim=20)
        self.dmp_goal = None

        self.recorder = Recorder(save_dir="/home/user/wam_ros/wam_ws/src/wam_haptic_dmps/dataset")
        self.episode_counter = self.recorder.get_next_episode_index()

        # Set up Joystick
        self.joy_fd = None
        self._init_joystick()

        self.kb_listener = keyboard.Listener(on_press=self._on_key_press)
        self.kb_listener.start()
        cprint("Initialization complete. Running teleop loop...", "green")
        cprint("Controls: [R] Start/Stop | [S] Save | [D] Discard", "cyan")

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
                    # if number == 1:  # 'o' button
                    #     self._handle_start_save_action()
                    # elif number == 0:  # 'x' button
                    #     self._handle_discard_action()
                    # SIXAXIS connection
                    if number == 13:  # 'o' button
                        print("press o")
                        self._handle_start_save_action()
                    elif number == 14:  # 'x' button
                        print("press x")
                        self._handle_discard_action()

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
                    if self.loop_state == "IDLE" or self.loop_state == "RECORDING":
                        self._handle_start_save_action()
                elif k == "s":
                    if self.loop_state == "PENDING":
                        self._save_episode()
                elif k == "d":
                    self._handle_discard_action()
        except Exception:
            pass

    def _handle_start_save_action(self):
        """Unified logic for 'r' / 's' on keyboard and 'o' on joystick."""
        if self.loop_state == "IDLE":
            self.loop_state = "RECORDING"
            cprint("\n[RECORDER] 🔴 EPISODE STARTED", "red", attrs=["bold"])
        elif self.loop_state == "RECORDING":
            self.loop_state = "PENDING"
            cprint("\n[RECORDER] ⏸ EPISODE PAUSED", "yellow")
            cprint("Press [S] or 'o' to Save, [D] or 'x' to Discard.", "cyan")
        elif self.loop_state == "PENDING":
            self._save_episode()

    def _handle_discard_action(self):
        """Unified logic for 'd' on keyboard and 'x' on joystick."""
        if self.loop_state in ["RECORDING", "PENDING"]:
            self.recorder.clear()
            self.loop_state = "IDLE"
            cprint(
                "\n[RECORDER] 🗑 Episode discarded. Press [R] or 'o' to start a new one.",
                "red",
            )

    def _save_episode(self):
        """Handles packaging and saving the episode to disk."""
        ep_name = f"episode_{self.episode_counter}"

        self.recorder.save_episode(ep_name)
        self.episode_counter += 1
        self.loop_state = "IDLE"
        cprint("[RECORDER] Ready for next episode. Press [R] or 'o' to start.", "cyan")

    def shutdown(self):
        cprint("Cleaning up streams and windows...", "red")
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
                    # Recording
                    if self.loop_state == "RECORDING":
                        self.recorder.add_low_dim_step(obs)

                precise_wait(t_cycle_end)
                iter_idx += 1


        except Exception as e:
            import traceback

            print("CRASHED WITH ERROR:")
            traceback.print_exc()

        finally:
            self.shutdown()

if __name__ == "__main__":
    rospy.init_node('dmp_learner')
    
    # Initialize
    learner = DMPLearner(remote_ip="127.0.0.1", recv_port=6554, DOF=7, hz=500)
    
    try:
        learner.run()
    except rospy.ROSInterruptException:
        pass
