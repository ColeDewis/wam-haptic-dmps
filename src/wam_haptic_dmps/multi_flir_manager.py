from harvesters.core import Harvester
from termcolor import cprint
import time
from wam_haptic_dmps.flir_streamer import FLIRStream


class MultiFLIRManager:
    def __init__(
        self,
        camera_configs: dict,
        gentl_path: str = "/opt/spinnaker/lib/spinnaker-gentl/Spinnaker_GenTL.cti",
    ):
        """
        camera_configs: dict mapping camera names to serial numbers,
        e.g., {"wrist": "18475182", "front": "18475176"}
        """
        cprint("Initializing Harvester System...", "green")
        self.harvester = Harvester()
        self.harvester.add_cti_file(gentl_path)
        self.harvester.update_device_info_list()

        self.streams = {}
        self.last_ids = {}
        for name, serial in camera_configs.items():
            self.streams[name] = FLIRStream(self.harvester, serial)
            self.last_ids[name] = 0

    def start_all(self):
        cprint("Starting all FLIR streams...", "green")
        for stream in self.streams.values():
            stream.start()
        # Give cameras a moment to warm up and fill buffers
        time.sleep(0.5)

    def read_all(self):
        """
        Reads from all cameras.
        Returns (True, {"wrist": img1, "front": img2}) if all succeeded.
        Returns (False, None) if ANY camera dropped a frame.
        """
        new_frames = {}
        for name, stream in self.streams.items():
            img, new_id, ts_ns = stream.read(self.last_ids[name])
            if img is not None:
                self.last_ids[name] = new_id
                new_frames[name] = (img, ts_ns)
        return new_frames

    def stop_all(self):
        cprint("Shutting down FLIR streams...", "red")
        for stream in self.streams.values():
            stream.running = False
            stream.stop()
        self.harvester.reset()

