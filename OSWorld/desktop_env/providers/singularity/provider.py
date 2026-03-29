import logging
import os
import platform
import time
import psutil
import requests
from filelock import FileLock
from pathlib import Path
import subprocess
from desktop_env.providers.base import Provider

logger = logging.getLogger("desktopenv.providers.singularity.SingularityProvider")
logger.setLevel(logging.INFO)

WAIT_TIME = 3
RETRY_INTERVAL = 1
LOCK_TIMEOUT = 300


class PortAllocationError(Exception):
    pass


class SingularityProvider(Provider):
    def __init__(self, region: str = None):
        super().__init__(region)
        # Check if singularity is available
        try:
            result = subprocess.run(["singularity", "--version"], capture_output=True, check=True)
            logger.info(f"Using Singularity: {result.stdout.decode().strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("Singularity not found! Please install Singularity to use this provider.")

        self.server_port = None
        self.vnc_port = None
        self.chromium_port = None
        self.vlc_port = None
        self.process = None
        self.environment = {"DISK_SIZE": "32G", "RAM_SIZE": "4G", "CPU_CORES": "4"}

        temp_dir = Path(os.getenv('TEMP') if platform.system() == 'Windows' else '/tmp')
        self.lock_file = temp_dir / "singularity_port_allocation.lck"
        self.port_registry_dir = temp_dir / "singularity_port_registry"
        self.port_registry_dir.mkdir(parents=True, exist_ok=True)
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)

        # Default SIF image path, can be overridden by environment variable
        self.sif_image = os.getenv("OSWORLD_SIF_IMAGE", "osworld-docker.sif")

    def _get_used_ports(self):
        """Get all currently used ports and reserved ports."""
        system_ports = set(conn.laddr.port for conn in psutil.net_connections())
        # Also check our internal registry for ports reserved by other processes
        reserved_ports = set()
        for p_file in self.port_registry_dir.glob("port_*"):
            try:
                reserved_ports.add(int(p_file.name.split("_")[1]))
            except:
                pass
        return system_ports | reserved_ports

    def _reserve_port(self, port):
        """Mark a port as reserved."""
        (self.port_registry_dir / f"port_{port}").touch()

    def _release_ports(self):
        """Release all ports reserved by this instance."""
        for port in [self.vnc_port, self.server_port, self.chromium_port, self.vlc_port]:
            if port:
                p_file = self.port_registry_dir / f"port_{port}"
                if p_file.exists():
                    try:
                        p_file.unlink()
                    except:
                        pass

    def _get_available_port(self, start_port: int) -> int:
        """Find next available port and reserve it."""
        used_ports = self._get_used_ports()
        port = start_port
        while port < 65354:
            if port not in used_ports:
                self._reserve_port(port)
                return port
            port += 1
        raise PortAllocationError(f"No available ports found starting from {start_port}")

    def _wait_for_vm_ready(self, timeout: int = 600):
        """Wait for VM to be ready by checking screenshot endpoint."""
        start_time = time.time()
        
        def check_screenshot():
            # Check if the process is still alive
            if self.process and self.process.poll() is not None:
                # Process has exited, read error output
                _, stderr = self.process.communicate()
                error_msg = stderr.decode() if stderr else "No error message"
                logger.error(f"Singularity process died. Error: {error_msg}")
                raise RuntimeError(f"Singularity process died: {error_msg}")

            try:
                response = requests.get(
                    f"http://localhost:{self.server_port}/screenshot",
                    timeout=(5, 5)
                )
                return response.status_code == 200
            except Exception:
                return False

        while time.time() - start_time < timeout:
            if check_screenshot():
                return True
            time.sleep(RETRY_INTERVAL)
        
        if self.process:
            logger.error(f"Timeout reached for port {self.server_port}. Checking process status...")
        
        raise TimeoutError(f"VM on port {self.server_port} failed to become ready within {timeout}s")

    def start_emulator(self, path_to_vm: str, headless: bool, os_type: str, name=None):
        # Use a single lock for all port allocation and container startup
        lock = FileLock(str(self.lock_file))
        
        try:
            with lock:
                # Add jitter to avoid simultaneous port scanning
                import random
                time.sleep(random.uniform(0, 3))

                # Allocate ports
                self.vnc_port = self._get_available_port(8006 + (os.getpid() % 100))
                self.server_port = self._get_available_port(5000 + (os.getpid() % 100))
                self.chromium_port = self._get_available_port(9222 + (os.getpid() % 100))
                self.vlc_port = self._get_available_port(8080 + (os.getpid() % 100))

                if not os.path.exists(self.sif_image):
                    raise FileNotFoundError(f"SIF image not found: {self.sif_image}")

                # Create temporary directories for system paths to allow writes
                temp_dir = Path(os.getenv('TEMP') if platform.system() == 'Windows' else '/tmp')
                run_dir = temp_dir / f"singularity_run_{os.getpid()}"
                run_dir.mkdir(parents=True, exist_ok=True)

                # Create a fake 'id' command to bypass root checks inside container
                fake_id_path = temp_dir / f"fake_id_{os.getpid()}"
                with open(fake_id_path, "w") as f:
                    f.write("#!/bin/sh\necho 0\n")
                os.chmod(fake_id_path, 0o755)

                # KVM acceleration is critical for performance
                kvm_flag = []
                if os.path.exists("/dev/kvm"):
                    kvm_flag = ["--bind", "/dev/kvm:/dev/kvm"]

                env = os.environ.copy()
                env.update(self.environment)
                # Singularity uses SINGULARITYENV_ prefix to pass vars into the container
                env.update({
                    "SINGULARITYENV_VNC_PORT": str(self.vnc_port),
                    "SINGULARITYENV_SERVER_PORT": str(self.server_port),
                    "SINGULARITYENV_CHROMIUM_PORT": str(self.chromium_port),
                    "SINGULARITYENV_VLC_PORT": str(self.vlc_port),
                    "VNC_PORT": str(self.vnc_port),
                    "SERVER_PORT": str(self.server_port),
                    "CHROMIUM_PORT": str(self.chromium_port),
                    "VLC_PORT": str(self.vlc_port),
                    "USER": "root", # Fake being root
                    "HOME": "/root" # Some scripts expect /root
                })

                cmd = [
                    "singularity", "run",
                    "--nv", 
                    "--writable-tmpfs", 
                    "--bind", f"{run_dir}:/run",       # Make /run writable
                    "--bind", f"{run_dir}:/var/run",   # Make /var/run writable
                    "--bind", f"{fake_id_path}:/usr/bin/id",
                    "--bind", f"{fake_id_path}:/bin/id",
                    *kvm_flag,
                    "--bind", f"{os.path.abspath(path_to_vm)}:/System.qcow2",
                    self.sif_image
                ]

                logger.info(f"Starting Singularity (Port {self.server_port}): {' '.join(cmd)}")
                self.process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid 
                )

                # Store for cleanup
                self.fake_id_path = fake_id_path
                self.run_dir = run_dir

            # Wait for VM to be ready
            self._wait_for_vm_ready()

            logger.info(f"Started Singularity container with ports - VNC: {self.vnc_port}, "
                       f"Server: {self.server_port}, Chrome: {self.chromium_port}, VLC: {self.vlc_port}")

            # Wait for VM to be ready
            self._wait_for_vm_ready()

        except Exception as e:
            logger.error(f"Error starting Singularity container: {e}")
            self.stop_emulator(path_to_vm)
            raise e

    def get_ip_address(self, path_to_vm: str) -> str:
        if not all([self.server_port, self.chromium_port, self.vnc_port, self.vlc_port]):
            raise RuntimeError("VM not started - ports not allocated")
        return f"localhost:{self.server_port}:{self.chromium_port}:{self.vnc_port}:{self.vlc_port}"

    def save_state(self, path_to_vm: str, snapshot_name: str):
        raise NotImplementedError("Snapshots not available for Singularity provider")

    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str):
        self.stop_emulator(path_to_vm)

    def stop_emulator(self, path_to_vm: str):
        if self.process:
            logger.info("Stopping Singularity container...")
            try:
                import signal
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=WAIT_TIME)
            except Exception as e:
                logger.error(f"Error stopping Singularity process: {e}")
                if self.process:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            finally:
                self.process = None
                self._release_ports() # Release reserved ports
                self.server_port = None
                self.vnc_port = None
                self.chromium_port = None
                self.vlc_port = None
    
    def pause_emulator(self):
        # Singularity doesn't have a direct pause command like Docker
        pass

    def unpause_emulator(self):
        pass
