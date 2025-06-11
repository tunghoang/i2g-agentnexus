"""
Production Monitoring
Comprehensive monitoring and health tracking for the platform
"""

import time
import json
import threading
import datetime
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

# Optional system monitoring
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from config.settings import MonitoringConfig
from agents.meta_agent import MetaAgent


class ProductionMonitor:
    """
    Production monitoring system for the Subsurface Data Platform

    Features:
    - Real-time health monitoring
    - Performance metrics collection
    - Error tracking and alerting
    - Automatic recovery detection
    - Dashboard data generation
    """

    def __init__(self, meta_agent: MetaAgent, servers: Dict[str, Any], config: MonitoringConfig):
        self.meta_agent = meta_agent
        self.servers = servers
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Monitoring state
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._start_time = time.time()

        # Metrics storage
        self.metrics_history: List[Dict[str, Any]] = []
        self.error_log: List[Dict[str, Any]] = []
        self.health_history: List[Dict[str, Any]] = []

        # Performance tracking
        self.performance_wrapper = None

    def start(self):
        """Start production monitoring"""
        if self._running:
            self.logger.warning("Production monitoring already running")
            return

        self.logger.info(" Starting production monitoring...")

        try:
            # Wrap meta agent for performance monitoring
            self._wrap_meta_agent()

            # Start background monitoring
            self._start_background_monitoring()

            # Log initial state
            self._log_initial_metrics()

            self._running = True
            self.logger.info("Production monitoring active")

        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
            raise

    def stop(self):
        """Stop production monitoring"""
        if not self._running:
            return

        self.logger.info("Stopping production monitoring...")

        self._running = False

        # Wait for monitor thread to finish
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)

        # Save final metrics
        self._save_metrics_snapshot()

        self.logger.info("Production monitoring stopped")

    def is_running(self) -> bool:
        """Check if monitoring is running"""
        return self._running and (self._monitor_thread is None or self._monitor_thread.is_alive())

    def _wrap_meta_agent(self):
        """Wrap meta agent for performance monitoring"""
        original_run = self.meta_agent.run

        def monitored_run(query: str) -> str:
            """Monitored version of meta agent run"""
            start_time = time.time()

            try:
                result = original_run(query)
                response_time = time.time() - start_time

                # Log successful query
                self._log_query_metrics(query, response_time, True, None)

                return result

            except Exception as e:
                response_time = time.time() - start_time

                # Log failed query
                self._log_query_metrics(query, response_time, False, str(e))

                raise

        self.meta_agent.run = monitored_run
        self.logger.debug("Meta agent wrapped for monitoring")

    def _start_background_monitoring(self):
        """Start background monitoring thread"""

        def monitor_loop():
            """Main monitoring loop"""
            while self._running:
                try:
                    # Collect system metrics
                    self._collect_system_metrics()

                    # Check server health
                    self._check_server_health()

                    # Check agent health
                    self._check_agent_health()

                    # Clean old metrics
                    self._cleanup_old_metrics()

                    # Save metrics snapshot
                    if len(self.metrics_history) % 10 == 0:  # Every 10 cycles
                        self._save_metrics_snapshot()

                except Exception as e:
                    self.logger.error(f"Monitoring loop error: {e}")

                # Wait for next cycle
                time.sleep(self.config.health_check_interval)

        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.name = "ProductionMonitor"
        self._monitor_thread.start()

        self.logger.debug("Background monitoring started")

    def _log_initial_metrics(self):
        """Log initial system state"""
        initial_metrics = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "monitoring_started",
            "platform_version": "2.0.0",
            "monitoring_config": {
                "health_check_interval": self.config.health_check_interval,
                "metrics_retention_days": self.config.metrics_retention_days
            }
        }

        self.metrics_history.append(initial_metrics)
        self.logger.info(f"Initial monitoring state logged")

    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            metrics = {
                "timestamp": datetime.datetime.now().isoformat(),
                "type": "system_metrics"
            }

            if PSUTIL_AVAILABLE:
                # System resources
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('.')

                metrics.update({
                    "cpu_percent": psutil.cpu_percent(interval=1),
                    "memory_percent": memory.percent,
                    "memory_available_gb": round(memory.available / (1024 ** 3), 2),
                    "disk_percent": disk.percent,
                    "disk_free_gb": round(disk.free / (1024 ** 3), 2),
                    "process_count": len(psutil.pids()),
                    "network_connections": len(psutil.net_connections())
                })
            else:
                metrics["note"] = "psutil not available - limited metrics"

            # Threading info
            metrics["active_threads"] = threading.active_count()

            # Platform uptime
            metrics["uptime_hours"] = round((time.time() - self._start_time) / 3600, 2)

            self.metrics_history.append(metrics)

        except Exception as e:
            self.logger.warning(f"System metrics collection failed: {e}")

    def _check_server_health(self):
        """Check health of all servers"""
        try:
            health_status = {
                "timestamp": datetime.datetime.now().isoformat(),
                "type": "server_health",
                "servers": {}
            }

            for server_name, server in self.servers.items():
                try:
                    if hasattr(server, 'is_ready') and hasattr(server, 'is_running'):
                        health_status["servers"][server_name] = {
                            "running": server.is_running(),
                            "ready": server.is_ready(),
                            "url": getattr(server, 'url', 'unknown')
                        }
                    else:
                        health_status["servers"][server_name] = {
                            "status": "unknown",
                            "note": "Server does not support health checking"
                        }

                except Exception as e:
                    health_status["servers"][server_name] = {
                        "status": "error",
                        "error": str(e)
                    }

            self.health_history.append(health_status)

            # Check for health issues
            unhealthy_servers = [
                name for name, status in health_status["servers"].items()
                if not status.get("ready", False) or status.get("status") == "error"
            ]

            if unhealthy_servers:
                self.logger.warning(f"Unhealthy servers detected: {unhealthy_servers}")

        except Exception as e:
            self.logger.warning(f"Server health check failed: {e}")

    def _check_agent_health(self):
        """Check health of meta agent"""
        try:
            agent_health = {
                "timestamp": datetime.datetime.now().isoformat(),
                "type": "agent_health"
            }

            # Get agent stats
            if hasattr(self.meta_agent, 'get_stats'):
                stats = self.meta_agent.get_stats()
                agent_health["stats"] = stats

            # Perform health check if available
            if hasattr(self.meta_agent, 'health_check'):
                health = self.meta_agent.health_check()
                agent_health["health"] = health

                if not health.get("functional", True):
                    self.logger.warning("Meta agent health check failed")

            self.health_history.append(agent_health)

        except Exception as e:
            self.logger.warning(f"Agent health check failed: {e}")

    def _log_query_metrics(self, query: str, response_time: float, success: bool, error: Optional[str]):
        """Log query performance metrics"""
        try:
            metrics = {
                "timestamp": datetime.datetime.now().isoformat(),
                "type": "query_metrics",
                "query_length": len(query),
                "response_time": round(response_time, 3),
                "success": success
            }

            if error:
                metrics["error"] = error

                # Also log to error log
                error_entry = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "query": query[:100] + "..." if len(query) > 100 else query,
                    "error": error,
                    "response_time": round(response_time, 3)
                }
                self.error_log.append(error_entry)

            self.metrics_history.append(metrics)

        except Exception as e:
            self.logger.warning(f"Query metrics logging failed: {e}")

    def _cleanup_old_metrics(self):
        """Clean up old metrics based on retention policy"""
        try:
            cutoff_time = datetime.datetime.now() - datetime.timedelta(days=self.config.metrics_retention_days)
            cutoff_iso = cutoff_time.isoformat()

            # Clean metrics history
            self.metrics_history = [
                m for m in self.metrics_history
                if m.get("timestamp", "") > cutoff_iso
            ]

            # Clean health history
            self.health_history = [
                h for h in self.health_history
                if h.get("timestamp", "") > cutoff_iso
            ]

            # Clean error log
            self.error_log = [
                e for e in self.error_log
                if e.get("timestamp", "") > cutoff_iso
            ]

        except Exception as e:
            self.logger.warning(f"Metrics cleanup failed: {e}")

    def _save_metrics_snapshot(self):
        """Save current metrics to file"""
        try:
            snapshot = {
                "timestamp": datetime.datetime.now().isoformat(),
                "platform_uptime_hours": round((time.time() - self._start_time) / 3600, 2),
                "metrics_count": len(self.metrics_history),
                "health_checks_count": len(self.health_history),
                "errors_count": len(self.error_log),
                "recent_metrics": self.metrics_history[-10:] if self.metrics_history else [],
                "recent_health": self.health_history[-5:] if self.health_history else [],
                "recent_errors": self.error_log[-10:] if self.error_log else []
            }

            # Save to monitoring directory
            monitoring_dir = Path("./monitoring")
            monitoring_dir.mkdir(exist_ok=True)

            snapshot_file = monitoring_dir / "metrics_snapshot.json"
            with open(snapshot_file, 'w') as f:
                json.dump(snapshot, f, indent=2)

            self.logger.debug(f"Metrics snapshot saved to {snapshot_file}")

        except Exception as e:
            self.logger.warning(f"Metrics snapshot save failed: {e}")

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Generate dashboard data for monitoring UI"""
        try:
            # Calculate summary statistics
            total_queries = len([m for m in self.metrics_history if m.get("type") == "query_metrics"])
            successful_queries = len([m for m in self.metrics_history
                                      if m.get("type") == "query_metrics" and m.get("success", False)])

            # Get latest system metrics
            latest_system = next((m for m in reversed(self.metrics_history)
                                  if m.get("type") == "system_metrics"), {})

            # Get latest server health
            latest_health = next((h for h in reversed(self.health_history)
                                  if h.get("type") == "server_health"), {})

            dashboard = {
                "timestamp": datetime.datetime.now().isoformat(),
                "platform_status": {
                    "uptime_hours": round((time.time() - self._start_time) / 3600, 2),
                    "monitoring_active": self.is_running(),
                    "total_queries": total_queries,
                    "success_rate": round((successful_queries / total_queries * 100), 2) if total_queries > 0 else 0,
                    "recent_errors": len(self.error_log[-24:])  # Last 24 errors
                },
                "system_metrics": {
                    "cpu_percent": latest_system.get("cpu_percent", "N/A"),
                    "memory_percent": latest_system.get("memory_percent", "N/A"),
                    "disk_percent": latest_system.get("disk_percent", "N/A"),
                    "active_threads": latest_system.get("active_threads", "N/A")
                },
                "server_health": latest_health.get("servers", {}),
                "recent_errors": self.error_log[-5:],  # Last 5 errors
                "metrics_retention": {
                    "total_metrics": len(self.metrics_history),
                    "retention_days": self.config.metrics_retention_days,
                    "next_cleanup": "Automatic"
                }
            }

            return dashboard

        except Exception as e:
            self.logger.error(f"Dashboard data generation failed: {e}")
            return {"error": str(e)}

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate detailed performance report"""
        try:
            query_metrics = [m for m in self.metrics_history if m.get("type") == "query_metrics"]

            if not query_metrics:
                return {"note": "No query metrics available"}

            # Calculate performance statistics
            response_times = [m["response_time"] for m in query_metrics if "response_time" in m]

            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                min_response_time = min(response_times)
                max_response_time = max(response_times)

                # Calculate percentiles
                sorted_times = sorted(response_times)
                p50 = sorted_times[len(sorted_times) // 2]
                p95 = sorted_times[int(len(sorted_times) * 0.95)]
            else:
                avg_response_time = min_response_time = max_response_time = p50 = p95 = 0

            # Success/failure analysis
            successful = len([m for m in query_metrics if m.get("success", False)])
            failed = len(query_metrics) - successful

            report = {
                "timestamp": datetime.datetime.now().isoformat(),
                "query_performance": {
                    "total_queries": len(query_metrics),
                    "successful_queries": successful,
                    "failed_queries": failed,
                    "success_rate_percent": round((successful / len(query_metrics) * 100), 2) if query_metrics else 0
                },
                "response_times": {
                    "average_seconds": round(avg_response_time, 3),
                    "min_seconds": round(min_response_time, 3),
                    "max_seconds": round(max_response_time, 3),
                    "p50_seconds": round(p50, 3),
                    "p95_seconds": round(p95, 3)
                },
                "error_analysis": {
                    "total_errors": len(self.error_log),
                    "recent_errors": len(self.error_log[-24:]),
                    "common_errors": self._analyze_common_errors()
                },
                "monitoring_info": {
                    "monitoring_uptime_hours": round((time.time() - self._start_time) / 3600, 2),
                    "metrics_collected": len(self.metrics_history),
                    "health_checks": len(self.health_history)
                }
            }

            return report

        except Exception as e:
            self.logger.error(f"Performance report generation failed: {e}")
            return {"error": str(e)}

    def _analyze_common_errors(self) -> List[Dict[str, Any]]:
        """Analyze common error patterns"""
        try:
            error_counts = {}

            for error_entry in self.error_log[-100:]:  # Last 100 errors
                error_msg = error_entry.get("error", "").lower()

                # Categorize errors
                if "rate limit" in error_msg or "429" in error_msg:
                    category = "rate_limit"
                elif "timeout" in error_msg:
                    category = "timeout"
                elif "file not found" in error_msg:
                    category = "file_not_found"
                elif "api" in error_msg and "key" in error_msg:
                    category = "api_key"
                elif "recursion" in error_msg:
                    category = "recursion_limit"
                else:
                    category = "other"

                error_counts[category] = error_counts.get(category, 0) + 1

            # Sort by frequency
            common_errors = [
                {"category": category, "count": count}
                for category, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
            ]

            return common_errors[:5]  # Top 5

        except Exception as e:
            self.logger.warning(f"Error analysis failed: {e}")
            return []


if __name__ == "__main__":
    # Test production monitor
    from config.settings import MonitoringConfig
    from agents.meta_agent import MetaAgent


    # Mock components for testing
    class MockMetaAgent:
        def run(self, query):
            time.sleep(0.1)  # Simulate processing time
            if "error" in query:
                raise Exception("Mock error")
            return f"Mock response to: {query}"

        def get_stats(self):
            return {"total_queries": 10, "uptime_hours": 1.0}

        def health_check(self):
            return {"functional": True}


    class MockServer:
        def is_running(self):
            return True

        def is_ready(self):
            return True

        @property
        def url(self):
            return "http://localhost:5000"


    # Test setup
    config = MonitoringConfig(health_check_interval=1)  # 1 second for testing
    meta_agent = MockMetaAgent()
    servers = {"test_server": MockServer()}

    monitor = ProductionMonitor(meta_agent, servers, config)

    print("Starting production monitor test...")
    monitor.start()

    # Test some queries
    print("Testing queries...")
    meta_agent.run("test query 1")
    meta_agent.run("test query 2")

    time.sleep(2)  # Let monitoring collect some data

    # Get dashboard data
    dashboard = monitor.get_dashboard_data()
    print(f"Dashboard data: {json.dumps(dashboard, indent=2)}")

    # Get performance report
    report = monitor.get_performance_report()
    print(f"Performance report: {json.dumps(report, indent=2)}")

    monitor.stop()
    print("Production monitor test completed")