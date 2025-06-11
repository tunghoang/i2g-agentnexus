"""
Base server management class
Common functionality for all server types
"""

import abc
import time
import threading
import logging
import requests
from typing import Optional


class BaseServer(abc.ABC):
    """
    Abstract base class for server management

    Provides common functionality for:
    - Server lifecycle management
    - Health checking
    - Status tracking
    - Clean shutdown
    """

    def __init__(self, name: str, host: str, port: int):
        self.name = name
        self.host = host
        self.port = port
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

        # State tracking
        self._server = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._ready = False
        self._start_time = None

    @property
    def url(self) -> str:
        """Get server URL"""
        return f"http://{self.host}:{self.port}"

    @abc.abstractmethod
    def _create_server(self):
        """Create the server instance - must be implemented by subclasses"""
        pass

    @abc.abstractmethod
    def _start_server(self):
        """Start the server - must be implemented by subclasses"""
        pass

    @abc.abstractmethod
    def _stop_server(self):
        """Stop the server - must be implemented by subclasses"""
        pass

    def start(self):
        """Start the server"""
        if self._running:
            self.logger.warning(f"{self.name} server already running")
            return

        self.logger.info(f"Starting {self.name} server on {self.url}")

        try:
            # Create server instance
            self._create_server()

            # Start server in background thread
            self._start_server()

            self._running = True
            self._start_time = time.time()

            self.logger.info(f"{self.name} server started successfully")

        except Exception as e:
            self.logger.error(f"Failed to start {self.name} server: {e}")
            self._running = False
            raise

    def stop(self):
        """Stop the server"""
        if not self._running:
            self.logger.warning(f"{self.name} server not running")
            return

        self.logger.info(f"Stopping {self.name} server...")

        try:
            self._stop_server()

            # Wait for thread to finish
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=10)

            self._running = False
            self._ready = False

            self.logger.info(f"{self.name} server stopped")

        except Exception as e:
            self.logger.error(f"Error stopping {self.name} server: {e}")
            raise

    def is_running(self) -> bool:
        """Check if server is running"""
        return self._running and (self._thread is None or self._thread.is_alive())

    def is_ready(self) -> bool:
        """Check if server is ready to accept requests"""
        return self._ready and self.is_running()

    def wait_ready(self, timeout: int = 30, check_interval: float = 1.0) -> bool:
        """
        Wait for server to be ready

        Args:
            timeout: Maximum time to wait in seconds
            check_interval: Time between checks in seconds

        Returns:
            True if server became ready, False if timeout
        """
        self.logger.info(f"Waiting for {self.name} server to be ready...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._check_health():
                self._ready = True
                self.logger.info(f"{self.name} server is ready")
                return True

            time.sleep(check_interval)

        self.logger.error(f"{self.name} server not ready after {timeout}s timeout")
        return False

    def _check_health(self) -> bool:
        """
        Check server health

        Returns:
            True if server is healthy, False otherwise
        """
        if not self.is_running():
            return False

        try:
            # Try to make a health check request
            response = requests.get(f"{self.url}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            # If no health endpoint, consider running thread as healthy
            return self.is_running()
        except Exception:
            return False

    def get_status(self) -> dict:
        """Get comprehensive server status"""
        uptime = time.time() - self._start_time if self._start_time else 0

        return {
            "name": self.name,
            "url": self.url,
            "running": self.is_running(),
            "ready": self.is_ready(),
            "uptime_seconds": round(uptime, 2),
            "thread_alive": self._thread.is_alive() if self._thread else False
        }

    def run_in_thread(self, target_func, *args, **kwargs):
        """
        Run a function in a background thread with error handling

        Args:
            target_func: Function to run in thread
            *args, **kwargs: Arguments for the function
        """

        def wrapped_target():
            try:
                self.logger.debug(f"Starting {self.name} server thread")
                target_func(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Error in {self.name} server thread: {e}")
                self._running = False
                raise

        self._thread = threading.Thread(
            target=wrapped_target,
            name=f"{self.name}ServerThread",
            daemon=True
        )
        self._thread.start()

        # Give thread a moment to start
        time.sleep(0.5)

        if not self._thread.is_alive():
            raise RuntimeError(f"{self.name} server thread failed to start")


class HealthCheckMixin:
    """Mixin for enhanced health checking"""

    def health_check_with_retry(self, max_retries: int = 3, retry_delay: float = 1.0) -> bool:
        """
        Health check with retry logic

        Args:
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds

        Returns:
            True if healthy, False otherwise
        """
        for attempt in range(max_retries + 1):
            if self._check_health():
                return True

            if attempt < max_retries:
                time.sleep(retry_delay)

        return False


if __name__ == "__main__":
    # Example implementation for testing
    class TestServer(BaseServer):
        def _create_server(self):
            self.logger.info("Creating test server")

        def _start_server(self):
            def dummy_server():
                while self._running:
                    time.sleep(1)

            self.run_in_thread(dummy_server)

        def _stop_server(self):
            self.logger.info("Stopping test server")


    # Test the base server
    server = TestServer("Test", "localhost", 8000)
    print(f"Server status: {server.get_status()}")

    server.start()
    print(f"Server running: {server.is_running()}")

    time.sleep(2)
    server.stop()
    print(f"Server status after stop: {server.get_status()}")