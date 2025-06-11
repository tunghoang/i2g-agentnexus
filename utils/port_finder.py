# utils/port_finder.py
"""
Port management utilities
"""

import socket
import logging
from typing import List


def find_available_port(start_port: int = 5000, max_tries: int = 20, host: str = 'localhost') -> int:
    """
    Find an available port starting from start_port

    Args:
        start_port: Starting port number to check
        max_tries: Maximum number of ports to try
        host: Host to bind to

    Returns:
        Available port number

    Raises:
        RuntimeError: If no available port found
    """
    logger = logging.getLogger(__name__)
    logger.debug(f" Searching for available port starting from {start_port}")

    for port in range(start_port, start_port + max_tries):
        try:
            # Try to create a socket on the port
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            sock.bind((host, port))
            sock.close()
            logger.debug(f"Found available port: {port}")
            return port
        except OSError as e:
            # Port is already in use, try the next one
            logger.debug(f"Port {port} is not available: {str(e)}")
            continue
        except Exception as e:
            logger.warning(f"Unexpected error checking port {port}: {str(e)}")
            continue

    # If we get here, no ports were available
    fallback_port = start_port + 1000  # Try a port well outside the normal range
    logger.warning(
        f"Could not find an available port in range {start_port}-{start_port + max_tries - 1}, "
        f"using fallback: {fallback_port}"
    )

    # Try the fallback port
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        sock.bind((host, fallback_port))
        sock.close()
        return fallback_port
    except Exception as e:
        logger.error(f"Failed to bind to fallback port {fallback_port}: {str(e)}")
        raise RuntimeError(f"No available ports found after trying {max_tries} ports and fallback")


def check_port_available(port: int, host: str = 'localhost') -> bool:
    """
    Check if a specific port is available

    Args:
        port: Port number to check
        host: Host to check on

    Returns:
        True if port is available, False otherwise
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        sock.bind((host, port))
        sock.close()
        return True
    except OSError:
        return False
    except Exception:
        return False


def find_available_ports(count: int, start_port: int = 5000) -> List[int]:
    """
    Find multiple available ports

    Args:
        count: Number of ports needed
        start_port: Starting port to search from

    Returns:
        List of available port numbers
    """
    ports = []
    current_port = start_port

    while len(ports) < count:
        try:
            port = find_available_port(current_port)
            ports.append(port)
            current_port = port + 1
        except RuntimeError:
            raise RuntimeError(f"Could not find {count} available ports starting from {start_port}")

    return ports


if __name__ == "__main__":
    # Test port finding
    print("Testing port finder...")

    # Find single port
    port = find_available_port(5000)
    print(f"Available port: {port}")

    # Check if port is available
    available = check_port_available(port)
    print(f"Port {port} available: {available}")

    # Find multiple ports
    ports = find_available_ports(3, 7000)
    print(f"Available ports: {ports}")