"""
production_segy_monitoring.py - Production monitoring and deployment system

This module provides comprehensive monitoring, health checks, performance optimization,
and deployment management for the production SEG-Y processing system.
"""

import os
import sys
import json
import time
import logging
import threading
import psutil
import traceback
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3
import schedule

logger = logging.getLogger(__name__)

@dataclass
class SystemHealthMetrics:
    """System health metrics for monitoring"""
    timestamp: str
    memory_usage_mb: float
    memory_percentage: float
    cpu_percentage: float
    disk_usage_gb: float
    disk_percentage: float
    active_processes: int
    
    # SEG-Y specific metrics
    files_processed_today: int
    files_failed_today: int
    average_processing_speed_mb_s: float
    total_data_processed_gb: float
    
    # Error tracking
    error_rate_percentage: float
    critical_errors: int
    warnings: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class SEGYSystemMonitor:
    """Production system monitoring for SEG-Y processing"""
    
    def __init__(self, db_path: str = "./monitoring/segy_metrics.db"):
        self.db_path = db_path
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Performance thresholds
        self.thresholds = {
            "memory_warning": 80.0,      # % memory usage
            "memory_critical": 90.0,
            "cpu_warning": 80.0,         # % CPU usage
            "cpu_critical": 95.0,
            "disk_warning": 85.0,        # % disk usage
            "disk_critical": 95.0,
            "error_rate_warning": 10.0,  # % error rate
            "error_rate_critical": 25.0,
            "processing_speed_min": 5.0   # MB/s minimum
        }
        
        self._initialize_database()
        
    def _initialize_database(self):
        """Initialize SQLite database for metrics storage"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    memory_usage_mb REAL,
                    memory_percentage REAL,
                    cpu_percentage REAL,
                    disk_usage_gb REAL,
                    disk_percentage REAL,
                    active_processes INTEGER,
                    files_processed_today INTEGER,
                    files_failed_today INTEGER,
                    average_processing_speed_mb_s REAL,
                    total_data_processed_gb REAL,
                    error_rate_percentage REAL,
                    critical_errors INTEGER,
                    warnings INTEGER
                )
            ''')
            
            # Create errors table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS error_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    component TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details TEXT,
                    resolved BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # Create processing log table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS processing_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    file_size_mb REAL,
                    processing_time_seconds REAL,
                    status TEXT NOT NULL,
                    error_message TEXT
                )
            ''')
            
            conn.commit()
            
    def collect_system_metrics(self) -> SystemHealthMetrics:
        """Collect current system health metrics"""
        try:
            # System metrics
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            disk = psutil.disk_usage('/')
            
            # Process metrics
            process_count = len(psutil.pids())
            
            # SEG-Y specific metrics from database
            today = datetime.now().strftime('%Y-%m-%d')
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get today's processing stats
                cursor.execute('''
                    SELECT 
                        COUNT(CASE WHEN status = 'success' THEN 1 END) as processed,
                        COUNT(CASE WHEN status = 'error' THEN 1 END) as failed,
                        AVG(CASE WHEN status = 'success' THEN file_size_mb/processing_time_seconds END) as avg_speed,
                        SUM(CASE WHEN status = 'success' THEN file_size_mb END) as total_gb
                    FROM processing_log 
                    WHERE date(timestamp) = ?
                ''', (today,))
                
                result = cursor.fetchone()
                files_processed = result[0] or 0
                files_failed = result[1] or 0
                avg_speed = result[2] or 0.0
                total_gb = (result[3] or 0.0) / 1024
                
                # Calculate error rate
                total_files = files_processed + files_failed
                error_rate = (files_failed / total_files * 100) if total_files > 0 else 0.0
                
                # Get recent critical errors and warnings
                cursor.execute('''
                    SELECT 
                        COUNT(CASE WHEN severity = 'ERROR' THEN 1 END) as critical,
                        COUNT(CASE WHEN severity = 'WARNING' THEN 1 END) as warnings
                    FROM error_log 
                    WHERE date(timestamp) = ? AND resolved = FALSE
                ''', (today,))
                
                error_result = cursor.fetchone()
                critical_errors = error_result[0] or 0
                warnings = error_result[1] or 0
                
            return SystemHealthMetrics(
                timestamp=datetime.now().isoformat(),
                memory_usage_mb=memory.used / 1024**2,
                memory_percentage=memory.percent,
                cpu_percentage=cpu_percent,
                disk_usage_gb=disk.used / 1024**3,
                disk_percentage=(disk.used / disk.total) * 100,
                active_processes=process_count,
                files_processed_today=files_processed,
                files_failed_today=files_failed,
                average_processing_speed_mb_s=avg_speed,
                total_data_processed_gb=total_gb,
                error_rate_percentage=error_rate,
                critical_errors=critical_errors,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
            # Return minimal metrics in case of error
            return SystemHealthMetrics(
                timestamp=datetime.now().isoformat(),
                memory_usage_mb=0, memory_percentage=0, cpu_percentage=0,
                disk_usage_gb=0, disk_percentage=0, active_processes=0,
                files_processed_today=0, files_failed_today=0,
                average_processing_speed_mb_s=0, total_data_processed_gb=0,
                error_rate_percentage=0, critical_errors=0, warnings=0
            )
            
    def store_metrics(self, metrics: SystemHealthMetrics):
        """Store metrics in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO system_metrics (
                        timestamp, memory_usage_mb, memory_percentage, cpu_percentage,
                        disk_usage_gb, disk_percentage, active_processes,
                        files_processed_today, files_failed_today, average_processing_speed_mb_s,
                        total_data_processed_gb, error_rate_percentage, critical_errors, warnings
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.timestamp, metrics.memory_usage_mb, metrics.memory_percentage,
                    metrics.cpu_percentage, metrics.disk_usage_gb, metrics.disk_percentage,
                    metrics.active_processes, metrics.files_processed_today, metrics.files_failed_today,
                    metrics.average_processing_speed_mb_s, metrics.total_data_processed_gb,
                    metrics.error_rate_percentage, metrics.critical_errors, metrics.warnings
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing metrics: {str(e)}")
            
    def check_health_thresholds(self, metrics: SystemHealthMetrics) -> List[Dict[str, Any]]:
        """Check if any health thresholds are exceeded"""
        alerts = []
        
        # Memory checks
        if metrics.memory_percentage > self.thresholds["memory_critical"]:
            alerts.append({
                "severity": "CRITICAL",
                "metric": "memory",
                "value": metrics.memory_percentage,
                "threshold": self.thresholds["memory_critical"],
                "message": f"Memory usage critical: {metrics.memory_percentage:.1f}%"
            })
        elif metrics.memory_percentage > self.thresholds["memory_warning"]:
            alerts.append({
                "severity": "WARNING",
                "metric": "memory",
                "value": metrics.memory_percentage,
                "threshold": self.thresholds["memory_warning"],
                "message": f"Memory usage high: {metrics.memory_percentage:.1f}%"
            })
            
        # CPU checks
        if metrics.cpu_percentage > self.thresholds["cpu_critical"]:
            alerts.append({
                "severity": "CRITICAL",
                "metric": "cpu",
                "value": metrics.cpu_percentage,
                "threshold": self.thresholds["cpu_critical"],
                "message": f"CPU usage critical: {metrics.cpu_percentage:.1f}%"
            })
        elif metrics.cpu_percentage > self.thresholds["cpu_warning"]:
            alerts.append({
                "severity": "WARNING",
                "metric": "cpu",
                "value": metrics.cpu_percentage,
                "threshold": self.thresholds["cpu_warning"],
                "message": f"CPU usage high: {metrics.cpu_percentage:.1f}%"
            })
            
        # Disk checks
        if metrics.disk_percentage > self.thresholds["disk_critical"]:
            alerts.append({
                "severity": "CRITICAL",
                "metric": "disk",
                "value": metrics.disk_percentage,
                "threshold": self.thresholds["disk_critical"],
                "message": f"Disk usage critical: {metrics.disk_percentage:.1f}%"
            })
        elif metrics.disk_percentage > self.thresholds["disk_warning"]:
            alerts.append({
                "severity": "WARNING",
                "metric": "disk",
                "value": metrics.disk_percentage,
                "threshold": self.thresholds["disk_warning"],
                "message": f"Disk usage high: {metrics.disk_percentage:.1f}%"
            })
            
        # Error rate checks
        if metrics.error_rate_percentage > self.thresholds["error_rate_critical"]:
            alerts.append({
                "severity": "CRITICAL",
                "metric": "error_rate",
                "value": metrics.error_rate_percentage,
                "threshold": self.thresholds["error_rate_critical"],
                "message": f"Error rate critical: {metrics.error_rate_percentage:.1f}%"
            })
        elif metrics.error_rate_percentage > self.thresholds["error_rate_warning"]:
            alerts.append({
                "severity": "WARNING",
                "metric": "error_rate",
                "value": metrics.error_rate_percentage,
                "threshold": self.thresholds["error_rate_warning"],
                "message": f"Error rate high: {metrics.error_rate_percentage:.1f}%"
            })
            
        # Processing speed checks
        if (metrics.average_processing_speed_mb_s > 0 and 
            metrics.average_processing_speed_mb_s < self.thresholds["processing_speed_min"]):
            alerts.append({
                "severity": "WARNING",
                "metric": "processing_speed",
                "value": metrics.average_processing_speed_mb_s,
                "threshold": self.thresholds["processing_speed_min"],
                "message": f"Processing speed low: {metrics.average_processing_speed_mb_s:.1f} MB/s"
            })
            
        return alerts
        
    def log_error(self, severity: str, component: str, message: str, details: str = None):
        """Log an error to the monitoring system"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO error_log (timestamp, severity, component, message, details)
                    VALUES (?, ?, ?, ?, ?)
                ''', (datetime.now().isoformat(), severity, component, message, details))
                conn.commit()
        except Exception as e:
            logger.error(f"Error logging to monitoring database: {str(e)}")
            
    def log_processing_result(self, operation: str, file_name: str, file_size_mb: float,
                             processing_time: float, status: str, error_message: str = None):
        """Log processing result for monitoring"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO processing_log (
                        timestamp, operation, file_name, file_size_mb, 
                        processing_time_seconds, status, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(), operation, file_name, file_size_mb,
                    processing_time, status, error_message
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error logging processing result: {str(e)}")
            
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            metrics = self.collect_system_metrics()
            alerts = self.check_health_thresholds(metrics)
            
            # Determine overall system status
            critical_alerts = [a for a in alerts if a["severity"] == "CRITICAL"]
            warning_alerts = [a for a in alerts if a["severity"] == "WARNING"]
            
            if critical_alerts:
                overall_status = "CRITICAL"
            elif warning_alerts:
                overall_status = "WARNING"
            else:
                overall_status = "HEALTHY"
                
            return {
                "overall_status": overall_status,
                "timestamp": metrics.timestamp,
                "metrics": metrics.to_dict(),
                "alerts": alerts,
                "summary": {
                    "critical_alerts": len(critical_alerts),
                    "warning_alerts": len(warning_alerts),
                    "files_processed_today": metrics.files_processed_today,
                    "files_failed_today": metrics.files_failed_today,
                    "current_error_rate": metrics.error_rate_percentage
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {
                "overall_status": "ERROR",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "alerts": [],
                "summary": {}
            }
            
    def start_monitoring(self, interval_seconds: int = 300):
        """Start continuous monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring is already active")
            return
            
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    metrics = self.collect_system_metrics()
                    self.store_metrics(metrics)
                    
                    # Check for alerts
                    alerts = self.check_health_thresholds(metrics)
                    for alert in alerts:
                        self.log_error(
                            alert["severity"], 
                            "system_monitor", 
                            alert["message"]
                        )
                        
                    logger.debug(f"Monitoring cycle completed - Status: {metrics.memory_percentage:.1f}% memory, {metrics.cpu_percentage:.1f}% CPU")
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {str(e)}")
                    
                time.sleep(interval_seconds)
                
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"System monitoring started (interval: {interval_seconds}s)")
        
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        logger.info("System monitoring stopped")

class SEGYDeploymentManager:
    """Production deployment and maintenance manager"""
    
    def __init__(self, config_dir: str = "./config", data_dir: str = "./data"):
        self.config_dir = Path(config_dir)
        self.data_dir = Path(data_dir)
        self.monitor = SEGYSystemMonitor()
        
    def perform_health_check(self) -> Dict[str, Any]:
        """Comprehensive system health check"""
        health_check = {
            "timestamp": datetime.now().isoformat(),
            "overall_health": "UNKNOWN",
            "checks": {}
        }
        
        try:
            # 1. System resources check
            system_status = self.monitor.get_system_status()
            health_check["checks"]["system_resources"] = {
                "status": system_status["overall_status"],
                "details": system_status["summary"]
            }
            
            # 2. Directory structure check
            required_dirs = [self.config_dir, self.data_dir, Path("./templates"), Path("./logs")]
            missing_dirs = [d for d in required_dirs if not d.exists()]
            
            health_check["checks"]["directory_structure"] = {
                "status": "PASS" if not missing_dirs else "FAIL",
                "missing_directories": [str(d) for d in missing_dirs]
            }
            
            # 3. Configuration files check
            config_files = list(self.config_dir.glob("*.yaml")) + list(self.config_dir.glob("*.json"))
            health_check["checks"]["configuration"] = {
                "status": "PASS" if config_files else "WARNING",
                "config_files_found": len(config_files),
                "files": [str(f) for f in config_files]
            }
            
            # 4. Template files check
            template_dir = Path("./templates")
            template_files = list(template_dir.glob("*.sgyfmt")) if template_dir.exists() else []
            health_check["checks"]["templates"] = {
                "status": "PASS" if template_files else "WARNING",
                "template_files_found": len(template_files),
                "files": [f.name for f in template_files]
            }
            
            # 5. Data directory check
            if self.data_dir.exists():
                segy_files = (list(self.data_dir.glob("*.sgy")) + 
                             list(self.data_dir.glob("*.segy")) +
                             list(self.data_dir.glob("**/*.sgy")) +
                             list(self.data_dir.glob("**/*.segy")))
                             
                total_size_gb = sum(f.stat().st_size for f in segy_files) / 1024**3
                
                health_check["checks"]["data_files"] = {
                    "status": "PASS",
                    "segy_files_found": len(segy_files),
                    "total_size_gb": round(total_size_gb, 2)
                }
            else:
                health_check["checks"]["data_files"] = {
                    "status": "WARNING",
                    "message": "Data directory does not exist"
                }
                
            # 6. Database connectivity check
            try:
                self.monitor.collect_system_metrics()
                health_check["checks"]["database"] = {"status": "PASS"}
            except Exception as e:
                health_check["checks"]["database"] = {
                    "status": "FAIL",
                    "error": str(e)
                }
                
            # Determine overall health
            check_statuses = [check.get("status", "UNKNOWN") for check in health_check["checks"].values()]
            
            if "FAIL" in check_statuses:
                health_check["overall_health"] = "CRITICAL"
            elif "WARNING" in check_statuses:
                health_check["overall_health"] = "WARNING"  
            elif all(status == "PASS" for status in check_statuses):
                health_check["overall_health"] = "HEALTHY"
            else:
                health_check["overall_health"] = "UNKNOWN"
                
        except Exception as e:
            health_check["overall_health"] = "ERROR"
            health_check["error"] = str(e)
            health_check["details"] = traceback.format_exc()
            
        return health_check
        
    def setup_production_environment(self) -> Dict[str, Any]:
        """Set up production environment with all required components"""
        setup_results = {
            "timestamp": datetime.now().isoformat(),
            "status": "IN_PROGRESS",
            "steps_completed": [],
            "errors": [],
            "warnings": []
        }
        
        try:
            # 1. Create directory structure
            directories = [
                Path("./config"),
                Path("./templates"), 
                Path("./data"),
                Path("./logs"),
                Path("./monitoring"),
                Path("./backups")
            ]
            
            for directory in directories:
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                    setup_results["steps_completed"].append(f"Created directory: {directory}")
                except Exception as e:
                    setup_results["errors"].append(f"Failed to create {directory}: {str(e)}")
                    
            # 2. Create default configuration
            try:
                from production_segy_multifile import create_default_config
                config_path = create_default_config("./config/segy_config.yaml")
                setup_results["steps_completed"].append(f"Created default config: {config_path}")
            except Exception as e:
                setup_results["errors"].append(f"Failed to create default config: {str(e)}")
                
            # 3. Initialize templates
            try:
                from production_segy_tools import TemplateValidator
                template_validator = TemplateValidator()
                default_template = template_validator.create_default_template("./templates")
                setup_results["steps_completed"].append(f"Created default template: {default_template}")
            except Exception as e:
                setup_results["errors"].append(f"Failed to create templates: {str(e)}")
                
            # 4. Initialize monitoring database
            try:
                monitor = SEGYSystemMonitor("./monitoring/segy_metrics.db")
                setup_results["steps_completed"].append("Initialized monitoring database")
            except Exception as e:
                setup_results["errors"].append(f"Failed to initialize monitoring: {str(e)}")
                
            # 5. Create sample configuration files
            try:
                sample_configs = {
                    "./config/production.yaml": {
                        "max_concurrent_files": 2,
                        "max_memory_gb": 8.0,
                        "enable_monitoring": True,
                        "log_level": "INFO"
                    },
                    "./config/development.yaml": {
                        "max_concurrent_files": 1,
                        "max_memory_gb": 4.0,
                        "enable_monitoring": False,
                        "log_level": "DEBUG"
                    }
                }
                
                for config_file, config_data in sample_configs.items():
                    with open(config_file, 'w') as f:
                        import yaml
                        yaml.dump(config_data, f, default_flow_style=False)
                    setup_results["steps_completed"].append(f"Created sample config: {config_file}")
                    
            except Exception as e:
                setup_results["warnings"].append(f"Failed to create sample configs: {str(e)}")
                
            # 6. Set up log rotation
            try:
                import logging.handlers
                
                # Configure rotating log handler
                log_file = "./logs/segy_production.log"
                handler = logging.handlers.RotatingFileHandler(
                    log_file, maxBytes=50*1024*1024, backupCount=5
                )
                handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                ))
                
                # Add to logger
                logger.addHandler(handler)
                setup_results["steps_completed"].append("Configured log rotation")
                
            except Exception as e:
                setup_results["warnings"].append(f"Failed to set up log rotation: {str(e)}")
                
            # Final status
            if setup_results["errors"]:
                setup_results["status"] = "COMPLETED_WITH_ERRORS"
            elif setup_results["warnings"]:
                setup_results["status"] = "COMPLETED_WITH_WARNINGS"
            else:
                setup_results["status"] = "COMPLETED_SUCCESSFULLY"
                
        except Exception as e:
            setup_results["status"] = "FAILED"
            setup_results["errors"].append(f"Setup failed: {str(e)}")
            setup_results["details"] = traceback.format_exc()
            
        return setup_results
        
    def generate_deployment_report(self) -> str:
        """Generate comprehensive deployment report"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("SEG-Y PRODUCTION SYSTEM DEPLOYMENT REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Health check
        health_check = self.perform_health_check()
        report_lines.append(f"OVERALL SYSTEM HEALTH: {health_check['overall_health']}")
        report_lines.append("")
        
        # Detailed checks
        report_lines.append("COMPONENT STATUS:")
        report_lines.append("-" * 40)
        for component, details in health_check["checks"].items():
            status = details.get("status", "UNKNOWN")
            report_lines.append(f"{component.replace('_', ' ').title()}: {status}")
            
            if "error" in details:
                report_lines.append(f"  Error: {details['error']}")
            elif "details" in details:
                for key, value in details["details"].items():
                    report_lines.append(f"  {key}: {value}")
                    
        report_lines.append("")
        
        # System metrics
        try:
            metrics = self.monitor.collect_system_metrics()
            report_lines.append("CURRENT SYSTEM METRICS:")
            report_lines.append("-" * 40)
            report_lines.append(f"Memory Usage: {metrics.memory_percentage:.1f}% ({metrics.memory_usage_mb:.0f} MB)")
            report_lines.append(f"CPU Usage: {metrics.cpu_percentage:.1f}%")
            report_lines.append(f"Disk Usage: {metrics.disk_percentage:.1f}% ({metrics.disk_usage_gb:.1f} GB)")
            report_lines.append(f"Files Processed Today: {metrics.files_processed_today}")
            report_lines.append(f"Processing Speed: {metrics.average_processing_speed_mb_s:.1f} MB/s")
            report_lines.append("")
        except Exception as e:
            report_lines.append(f"Could not collect system metrics: {str(e)}")
            report_lines.append("")
            
        # Recommendations
        report_lines.append("RECOMMENDATIONS:")
        report_lines.append("-" * 40)
        
        if health_check["overall_health"] == "CRITICAL":
            report_lines.append("❌ CRITICAL ISSUES DETECTED - IMMEDIATE ACTION REQUIRED")
            report_lines.append("   - Review failed components above")
            report_lines.append("   - Check system resources")
            report_lines.append("   - Verify file permissions and accessibility")
        elif health_check["overall_health"] == "WARNING":
            report_lines.append("⚠️ WARNINGS DETECTED - REVIEW RECOMMENDED")
            report_lines.append("   - Address warning conditions when possible")
            report_lines.append("   - Monitor system performance")
        else:
            report_lines.append("✅ SYSTEM HEALTHY - READY FOR PRODUCTION")
            report_lines.append("   - Start monitoring if not already active")
            report_lines.append("   - Begin regular maintenance schedule")

        report_lines.append("")
        report_lines.append("NEXT STEPS:")
        report_lines.append("-" * 40)
        report_lines.append("1. Address any critical or warning issues")
        report_lines.append("2. Start system monitoring: monitor.start_monitoring()")
        report_lines.append("3. Test with sample SEG-Y files")
        report_lines.append("4. Set up automated backup schedule")
        report_lines.append("5. Configure alerting for critical conditions")

        report_lines.append("")
        report_lines.append("=" * 80)

        return "\n".join(report_lines)

# Production deployment functions
def deploy_production_segy_system(config_dir: str = "./config", data_dir: str = "./data") -> Dict[str, Any]:
    """Complete production deployment of SEG-Y system"""
    logger.info("Starting production SEG-Y system deployment")

    deployment_manager = SEGYDeploymentManager(config_dir, data_dir)

    # Run setup
    setup_results = deployment_manager.setup_production_environment()

    # Run health check
    health_check = deployment_manager.perform_health_check()

    # Generate report
    report = deployment_manager.generate_deployment_report()

    deployment_result = {
        "deployment_status": setup_results["status"],
        "deployment_details": setup_results,
        "health_check": health_check,
        "deployment_report": report,
        "timestamp": datetime.now().isoformat()
    }

    logger.info(f"Production deployment completed with status: {setup_results['status']}")

    return deployment_result

def get_production_system_status() -> Dict[str, Any]:
    """Get current production system status"""
    monitor = SEGYSystemMonitor()
    return monitor.get_system_status()

def start_production_monitoring(interval_seconds: int = 300):
    """Start production system monitoring"""
    monitor = SEGYSystemMonitor()
    monitor.start_monitoring(interval_seconds)
    logger.info(f"Production monitoring started with {interval_seconds}s interval")
    return monitor

# Maintenance automation
def schedule_maintenance_tasks():
    """Schedule automated maintenance tasks"""

    def daily_cleanup():
        """Daily maintenance tasks"""
        logger.info("Running daily maintenance tasks")

        # Clean old log files
        log_dir = Path("./logs")
        if log_dir.exists():
            cutoff_date = datetime.now() - timedelta(days=30)
            for log_file in log_dir.glob("*.log*"):
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    try:
                        log_file.unlink()
                        logger.info(f"Cleaned old log file: {log_file}")
                    except Exception as e:
                        logger.error(f"Failed to clean log file {log_file}: {e}")

        # Compact monitoring database
        try:
            monitor = SEGYSystemMonitor()
            with sqlite3.connect(monitor.db_path) as conn:
                conn.execute("VACUUM")
                logger.info("Compacted monitoring database")
        except Exception as e:
            logger.error(f"Failed to compact database: {e}")

    def weekly_health_check():
        """Weekly comprehensive health check"""
        logger.info("Running weekly health check")

        try:
            deployment_manager = SEGYDeploymentManager()
            health_check = deployment_manager.perform_health_check()

            # Log results
            if health_check["overall_health"] in ["CRITICAL", "WARNING"]:
                logger.warning(f"Weekly health check: {health_check['overall_health']}")
                for component, details in health_check["checks"].items():
                    if details.get("status") in ["FAIL", "WARNING"]:
                        logger.warning(f"  {component}: {details}")
            else:
                logger.info("Weekly health check: System healthy")

        except Exception as e:
            logger.error(f"Weekly health check failed: {e}")

    # Schedule tasks
    schedule.every().day.at("02:00").do(daily_cleanup)
    schedule.every().monday.at("03:00").do(weekly_health_check)

    logger.info("Maintenance tasks scheduled")