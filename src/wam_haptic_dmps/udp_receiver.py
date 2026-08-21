import socket
import struct


class UDPReceiver:
    def __init__(self, ip, recv_port, dof=7):
        """
        :param recv_port: The port THIS Python code listens on.
        :param dof: Degrees of freedom (default 7).
        """
        self.dof = dof
        self.recv_port = recv_port

        # 13 DOF-length arrays:
        #   follower_jp, follower_jv, follower_dyngravcomp_torque,
        #   environment_torque, filtered_environment_torque,
        #   leader_jp, leader_jv, leader_dyngravcomp_torque,
        #   human_torque, filtered_human_torque,
        #   policyJp, policyJt, policyTorqueScale
        # + follower(cart_pos 3 + quat 4) + leader(cart_pos 3 + quat 4)
        # + gripper_pos + gripper_vel + gripper_torque
        # 'Q'  = time_to_chunk_end (ns)
        # 'Q'  = uint64_t (timestamp)
        # bytes: 880        

        num_doubles = (13 * self.dof) + 17
        self.fmt = f"<{num_doubles}dQQ"
        self.packet_size = struct.calcsize(self.fmt)

        # Receiver Socket
        self.sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock_recv.bind((ip, self.recv_port))
            self.sock_recv.setblocking(False)
            print(
                f"UDP [Recorder]: Listening on port {self.recv_port} for {self.packet_size}-byte packets"
            )
        except OSError as e:
            print(f"Error binding to port {self.recv_port}: {e}")
            self.sock_recv = None

    def receive_latest_data(self):
        """
        Drains the OS UDP buffer and returns ONLY the most recent packet.
        This ensures Python doesn't fall behind the high-frequency C++ control loop.
        """
        if not self.sock_recv:
            return None

        latest_data = None

        while True:
            try:
                data, _ = self.sock_recv.recvfrom(1024)
                if len(data) == self.packet_size:
                    latest_data = data
                else:
                    print(f"⚠️ Dropping packet! Expected {self.packet_size} bytes, got {len(data)} bytes.")
            except BlockingIOError:
                break  # Buffer is empty
            except Exception as e:
                print(f"Receive Error: {e}")
                break

        # If we didn't get any valid data this loop, return None
        if latest_data is None:
            return None

        # Unpack the freshest packet
        unpacked = struct.unpack(self.fmt, latest_data)

        dof = self.dof

        idx_follower_jp = 0
        idx_follower_jv = dof
        idx_follower_dyngravcomp_torque = dof * 2
        idx_environment_torque = dof * 3
        idx_filtered_environment_torque = dof * 4
        idx_leader_jp = dof * 5
        idx_leader_jv = dof * 6
        idx_leader_dyngravcomp_torque = dof * 7
        idx_human_torque = dof * 8
        idx_filtered_human_torque = dof * 9
        idx_policy_jp = dof * 10
        idx_policy_jt = dof * 11
        idx_policy_torque_scale = dof * 12
        idx_follower_cart_pos = dof * 13
        idx_follower_quat = dof * 13 + 3
        idx_leader_cart_pos = dof * 13 + 7
        idx_leader_quat = dof * 13 + 10
        idx_gripper_pos = dof * 13 + 14
        idx_gripper_vel = dof * 13 + 15
        idx_gripper_torque = dof * 13 + 16
        idx_time_to_chunk_end = dof * 13 + 17
        idx_timestamp = dof * 13 + 18
 
        return {
            "follower_jp": list(unpacked[idx_follower_jp:idx_follower_jv]),
            "follower_jv": list(unpacked[idx_follower_jv:idx_follower_dyngravcomp_torque]),
            "follower_dyngravcomp_torque": list(unpacked[idx_follower_dyngravcomp_torque:idx_environment_torque]),
            "environment_torque": list(unpacked[idx_environment_torque:idx_filtered_environment_torque]),
            "filtered_environment_torque": list(unpacked[idx_filtered_environment_torque:idx_leader_jp]),
            "leader_jp": list(unpacked[idx_leader_jp:idx_leader_jv]),
            "leader_jv": list(unpacked[idx_leader_jv:idx_leader_dyngravcomp_torque]),
            "leader_dyngravcomp_torque": list(unpacked[idx_leader_dyngravcomp_torque:idx_human_torque]),
            "human_torque": list(unpacked[idx_human_torque:idx_filtered_human_torque]),
            "filtered_human_torque": list(unpacked[idx_filtered_human_torque:idx_policy_jp]),
            "policy_jp": list(unpacked[idx_policy_jp:idx_policy_jt]),
            "policy_jt": list(unpacked[idx_policy_jt:idx_policy_torque_scale]),
            "policy_torque_scale": list(unpacked[idx_policy_torque_scale:idx_follower_cart_pos]),
            "follower_cart_pos": list(unpacked[idx_follower_cart_pos:idx_follower_quat]),
            "follower_quat_wxyz": list(unpacked[idx_follower_quat:idx_leader_cart_pos]),
            "leader_cart_pos": list(unpacked[idx_leader_cart_pos:idx_leader_quat]),
            "leader_quat_wxyz": list(unpacked[idx_leader_quat:idx_gripper_pos]),
            "gripper_pos": unpacked[idx_gripper_pos],
            "gripper_vel": unpacked[idx_gripper_vel],
            "gripper_torque": unpacked[idx_gripper_torque],
            "time_to_chunk_end_ns": unpacked[idx_time_to_chunk_end],
            "timestamp_ns": unpacked[idx_timestamp],
        }



    def close(self):
        if self.sock_recv:
            self.sock_recv.close()


