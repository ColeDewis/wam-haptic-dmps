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
import cv2
from wam_haptic_dmps.udp_sender import UDPSender
from wam_haptic_dmps.udp_receiver import UDPReceiver
from wam_haptic_dmps.recorder import Recorder
from wam_haptic_dmps.multi_flir_manager import MultiFLIRManager
from wam_haptic_dmps.precise_sleep import precise_wait

def preprocess_image(
    img_bgr: np.ndarray, crop_scale: float = 0.9, out_size=(224, 224)
) -> np.ndarray:
    """Center-crop by area 'crop_scale' and resize to out_size using OpenCV. Keeps image in BGR."""
    H, W = img_bgr.shape[:2]
    s = float(crop_scale) ** 0.5
    crop_h, crop_w = int(round(H * s)), int(round(W * s))

    y0 = max((H - crop_h) // 2, 0)
    x0 = max((W - crop_w) // 2, 0)

    img_cropped = img_bgr[y0 : y0 + crop_h, x0 : x0 + crop_w]
    img_resized = cv2.resize(img_cropped, out_size, interpolation=cv2.INTER_AREA)

    return img_resized

class DMPRunner:
    """
    Controls:
      [R] / joystick button 1 -> Start recording a new episode
      [D] / joystick button 0 -> Stop recording, SAVE the episode, bump episode_counter
      [S] / joystick button 2 -> Select the just-saved episode as the DMP demo
                                  (dmp_idx = episode_counter - 1), train the DMP on it
    """

    def __init__(self, remote_ip="127.0.0.1", leader_send_port=10000, follower_send_port=20000, recv_port=6554, DOF=7, hz=10):
        self.horizon = 8  # we also send over the current state so on the receiving side we get +1 actions

        self.udp_receiver = UDPReceiver(
            remote_ip, recv_port, DOF
        )
        self.leader_udp_sender = UDPSender(
            # remote_ip, send_port, DOF=DOF, horizon=self.horizon + 1
            remote_ip, leader_send_port, DOF=DOF, horizon=self.horizon
        )
        self.follower_udp_sender = UDPSender(
            # remote_ip, send_port, DOF=DOF, horizon=self.horizon + 1
            remote_ip, follower_send_port, DOF=DOF, horizon=self.horizon
        )

        # FLIR setup
        camera_configs = {
            "wrist_image": "18475182",
            "front_image": "18475176",
        }
        self.camera_manager = MultiFLIRManager(camera_configs)
        self.camera_manager.start_all()


        self.send_interval = 0.4  # interval between sent action chunks
        self.last_send_time = 0.0

        self.inference_time = 1.0          # estimate of how long inf takes
        self.inference_time_alpha = 0.2    # EMA smoothing factor for the estimate

        self.loop_state = "IDLE"  # "IDLE" | "RECORDING"

        self.dof = DOF
        self.dt = 1 / hz
        self.dmp = DMP(n_dims=DOF + 1, dt=self.dt, n_weights_per_dim=20)
        self.dmp_goal = None
        self.dmp_output = None
        self.dmp_start_idx = 0
        self.dmp_idx = None  # index of episode used to train the DMP; None until selected

        self.recorder = Recorder(save_dir="/home/user/wam_ros/wam_ws/src/wam_haptic_dmps/dataset")
        self.episode_counter = self.recorder.get_next_episode_index()

        # Set up Joystick
        self.joy_fd = None
        self._init_joystick()

        self.kb_listener = keyboard.Listener(on_press=self._on_key_press)
        self.kb_listener.start()
        cprint("Initialization complete. Running teleop loop...", "green")
        cprint("Controls: [R]/o Start recording | [D]/x Stop+Save | [S]/up Select DMP demo", "cyan")

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

                if ev_type == 0x01 and value == 1:
                    # BLUETOOTHCTL connection
                    # if number == 1:  # 'o' button
                    #     print("press o")
                    #     self._handle_start_recording()
                    # elif number == 0:  # 'x' button
                    #     print("press x")
                    #     self._handle_stop_and_save()
                    # elif number == 8:   # hat Y axis, up = negative — confirm exact axis# and sign from your log!
                    #     print("press up")
                    #     self._handle_stop_and_save()
                    #     self._handle_select_dmp_demo()
                    # SIXAXIS connection
                    if number == 13:  # 'o' button
                        print("press o")
                        self._handle_start_recording()
                    elif number == 14:  # 'x' button
                        print("press x")
                        self._handle_stop_and_save()
                    elif number == 4:   # hat Y axis, up = negative — confirm exact axis# and sign from your log!
                        print("press up")
                        self._handle_stop_and_save()
                        self._handle_select_dmp_demo()

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
                    self._handle_start_recording()
                elif k == "d":
                    self._handle_stop_and_save()
                elif k == "s":
                    self._handle_stop_and_save()
                    self._handle_select_dmp_demo()
        except Exception:
            pass

    def _handle_start_recording(self):
        """'r' on keyboard / 'o' on joystick: begin recording a new episode."""
        if self.loop_state == "IDLE":
            self.recorder.clear()
            self.loop_state = "RECORDING"
            self.dmp_output = None
            self.dmp_start_idx = 0
            self.last_send_time = 0.0
            cprint("\n[RUNNER] \U0001F534 EPISODE STARTED", "red", attrs=["bold"])
            if self.dmp_idx is not None:
                cprint(f"[RUNNER] DMP demo active (episode_{self.dmp_idx}) -- will send actions while recording.", "cyan")
            else:
                cprint("[RUNNER] No DMP demo selected -- recording locally only, no UDP actions will be sent.", "yellow")

    def _handle_stop_and_save(self):
        """'d' on keyboard / 'x' on joystick: stop recording, save episode, stop any action sending."""
        if self.loop_state == "RECORDING":
            ep_name = f"episode_{self.episode_counter}"
            self.recorder.save_episode(ep_name)
            self.episode_counter += 1

            self.loop_state = "IDLE"
            self.dmp_output = None
            self.dmp_start_idx = 0

            cprint(f"\n[RUNNER] \U0001F4BE Saved {ep_name}. Ready for next episode.", "green")
            cprint("Press [R] or 'o' to start recording, [S] or 'up' to pick a DMP demo.", "cyan")

    def _handle_select_dmp_demo(self):
        """'s' on keyboard / 'up' on joystick: use the just-saved episode as the DMP demo."""
        selected_idx = self.episode_counter - 1
        trajectory_buffer = self.recorder.load_episode(selected_idx)

        cprint(f"[RUNNER] Training DMP on episode_{selected_idx}...", "cyan")
        n_steps = len(trajectory_buffer)
        execution_time = (n_steps - 1) * self.dt
        T = np.linspace(0, execution_time, n_steps)
        self.dmp.imitate(T, np.array(trajectory_buffer))
        self.dmp.set_execution_time_(execution_time)
        self.dmp_goal = np.array(trajectory_buffer[-1])

        self.dmp_idx = selected_idx
        self.dmp_output = None
        self.dmp_start_idx = 0
        cprint(f"[RUNNER] DMP demo set to episode_{self.dmp_idx}.", "green")

    def shutdown(self):
        cprint("Cleaning up streams and windows...", "red")
        self.leader_udp_sender.close()
        self.follower_udp_sender.close()
        self.udp_receiver.close()
        if self.joy_fd is not None:
            try:
                os.close(self.joy_fd)
            except Exception:
                pass
        self.camera_manager.stop_all()

        os._exit(0)

    def _read_images(self):
        status, raw_frames = self.camera_manager.read_all()
        if not status:
            return False, None

        # TODO: we really shouldn't be doing this here, it would make more sense
        # for the policy to preprocess itself.
        processed_frames = {}
        for name, img_bgr in raw_frames.items():
            proc_img = preprocess_image(img_bgr)
            processed_frames[name] = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)

        return True, processed_frames

    def _handle_new_frames(self, new_frames):
        for name, (img_bgr, ts_ns) in new_frames.items():
            proc = preprocess_image(img_bgr)
            proc = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
            self.recorder.add_image_step(name, proc, ts_ns)

    def _step_dmp_rollout(self, obs):
        """Runs the DMP rollout (if a demo is selected) and sends UDP action chunks while recording."""
        if self.dmp_output is None:
            # configure the DMP to start from current position, keeping the same end goal.
            self.dmp.configure(
                start_y=np.array([*obs["follower_jp"], obs["gripper_pos"]]),
                start_yd=np.array([*obs["follower_jv"], obs["gripper_vel"]]),
                goal_y=self.dmp_goal,
            )
            _, self.dmp_output = self.dmp.open_loop()

        # Once the rollout has been fully sent, stop sending further packets
        # (recording itself keeps going until the user hits D/x).
        if self.dmp_start_idx >= len(self.dmp_output):
            return

        dmp_end_idx = min(self.dmp_start_idx + self.horizon, len(self.dmp_output))

        action_chunk = self.dmp_output[self.dmp_start_idx:dmp_end_idx, :]

        if action_chunk.shape[0] < self.horizon:
            pad_count = self.horizon - action_chunk.shape[0]
            pad = np.tile(action_chunk[-1], (pad_count, 1))
            action_chunk = np.vstack([action_chunk, pad])

        if (time.time() - self.last_send_time) >= self.send_interval:
            time_to_skip_ns = obs["time_to_chunk_end_ns"]

            print(self.dmp_start_idx)
            print(action_chunk)
            print(action_chunk.shape)
            print(dmp_end_idx)

            self.leader_udp_sender.send_action_chunk(action_chunk, time_to_skip_ns)
            self.follower_udp_sender.send_action_chunk(action_chunk, time_to_skip_ns)
            self.dmp_start_idx += self.horizon

            if dmp_end_idx == len(self.dmp_output):
                cprint("[RUNNER] DMP rollout finished sending (recording continues).", "cyan")

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

                new_states = self.udp_receiver.receive_all_new()
                new_frames = self.camera_manager.read_all() # non blocking

                if self.loop_state == "RECORDING":
                    for state_dict in new_states:
                        self.recorder.add_low_dim_step(state_dict)
                    if new_frames:
                        self._handle_new_frames(new_frames)

                    if len(new_states) > 0:
                        time_to_chunk_end_s = new_states[-1]["time_to_chunk_end_ns"] / 1e9
                        interval_elapsed = (time.time() - self.last_send_time) >= self.send_interval

                        should_infer = (time_to_chunk_end_s <= self.inference_time) and interval_elapsed

                        if should_infer:
                            t_infer_start = time.monotonic()

                            # Only send UDP actions if a DMP demo has actually been selected.
                            if self.dmp_idx is not None:
                                self._step_dmp_rollout(new_states[-1])
                                
                            elapsed = time.monotonic() - t_infer_start
                            self.inference_time = (
                                self.inference_time_alpha * elapsed
                                + (1 - self.inference_time_alpha) * self.inference_time
                            )
                            print(f"inf time {self.inference_time}")

                            self.last_send_time = time.time()

                precise_wait(t_cycle_end)
                iter_idx += 1

        except Exception as e:
            import traceback

            print("CRASHED WITH ERROR:")
            traceback.print_exc()

        finally:
            self.shutdown()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)


    rospy.init_node('dmp_runner')

    runner = DMPRunner(remote_ip="127.0.0.1", leader_send_port=10000, follower_send_port=20000, recv_port=6554, DOF=7, hz=500)

    try:
        runner.run()
    except rospy.ROSInterruptException:
        pass
