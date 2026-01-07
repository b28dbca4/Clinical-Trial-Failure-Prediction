from pathlib import Path
import psutil
import time
from typing import Dict, Any
from dataclasses import dataclass
from datetime import datetime
import threading
import logging
import json

logger = logging.getLogger(__name__)
@dataclass
class SystemMetrics:
    """System metrics"""
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_io: Dict[str, float]
    process_memory_mb: float
    open_files: int
    
class ResourceMonitor:
    """System resource monitor"""
    
    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self.metrics_history = []
        self.monitoring = False
        self.monitor_thread = None
        self.start_time = None
        
    def start(self):
        """Start monitoring"""
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Monitoring loop"""
        while self.monitoring:
            metrics = self.collect_metrics()
            self.metrics_history.append(metrics)
            time.sleep(self.interval)
    
    def collect_metrics(self) -> SystemMetrics:
        """Collect system metrics"""
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024**3)
        
        # Disk
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        disk_free_gb = disk.free / (1024**3)
        
        # Network
        net_io = psutil.net_io_counters()
        network_io = {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv
        }
        
        # Process info
        process = psutil.Process()
        process_memory_mb = process.memory_info().rss / (1024**2)
        open_files = len(process.open_files())
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_gb=memory_used_gb,
            disk_usage_percent=disk_percent,
            disk_free_gb=disk_free_gb,
            network_io=network_io,
            process_memory_mb=process_memory_mb,
            open_files=open_files
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of monitoring session"""
        if not self.metrics_history:
            return {}
        
        # Calculate statistics
        cpu_values = [m.cpu_percent for m in self.metrics_history]
        memory_values = [m.memory_percent for m in self.metrics_history]
        
        duration = time.time() - self.start_time
        
        return {
            "monitoring_duration_seconds": duration,
            "samples_collected": len(self.metrics_history),
            "cpu_stats": {
                "mean": sum(cpu_values) / len(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values)
            },
            "memory_stats": {
                "mean": sum(memory_values) / len(memory_values),
                "max": max(memory_values),
                "min": min(memory_values)
            },
            "final_metrics": self._metrics_to_dict(self.metrics_history[-1]),
            "peak_memory_mb": max(m.process_memory_mb for m in self.metrics_history)
        }
    
    def _metrics_to_dict(self, metrics: SystemMetrics) -> Dict:
        """Convert metrics to dict"""
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": metrics.cpu_percent,
            "memory_percent": metrics.memory_percent,
            "memory_used_gb": metrics.memory_used_gb,
            "disk_usage_percent": metrics.disk_usage_percent,
            "disk_free_gb": metrics.disk_free_gb,
            "process_memory_mb": metrics.process_memory_mb,
            "open_files": metrics.open_files
        }
    
    def save_report(self, output_path: Path):
        """Save monitoring report"""
        summary = self.get_summary()
        
        report = {
            "monitoring_session": {
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "duration_seconds": summary.get("monitoring_duration_seconds", 0),
                "samples_collected": summary.get("samples_collected", 0)
            },
            "resource_usage_summary": summary,
            "metrics_history": [self._metrics_to_dict(m) for m in self.metrics_history]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Monitoring report saved to {output_path}")
        
        # Can also visualize
        self._create_visualization(output_path.parent / "monitoring_plots.png")
    
    def _create_visualization(self, output_path: Path):
        """Create visualization of metrics"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # CPU usage
        timestamps = range(len(self.metrics_history))
        cpu_values = [m.cpu_percent for m in self.metrics_history]
        axes[0, 0].plot(timestamps, cpu_values)
        axes[0, 0].set_title('CPU Usage (%)')
        axes[0, 0].set_xlabel('Sample')
        axes[0, 0].set_ylabel('CPU %')
        axes[0, 0].grid(True)
        
        # Memory usage
        memory_values = [m.memory_percent for m in self.metrics_history]
        axes[0, 1].plot(timestamps, memory_values, color='orange')
        axes[0, 1].set_title('Memory Usage (%)')
        axes[0, 1].set_xlabel('Sample')
        axes[0, 1].set_ylabel('Memory %')
        axes[0, 1].grid(True)
        
        # Process memory
        process_memory = [m.process_memory_mb for m in self.metrics_history]
        axes[1, 0].plot(timestamps, process_memory, color='green')
        axes[1, 0].set_title('Process Memory (MB)')
        axes[1, 0].set_xlabel('Sample')
        axes[1, 0].set_ylabel('Memory (MB)')
        axes[1, 0].grid(True)
        
        # Disk usage
        disk_values = [m.disk_usage_percent for m in self.metrics_history]
        axes[1, 1].plot(timestamps, disk_values, color='red')
        axes[1, 1].set_title('Disk Usage (%)')
        axes[1, 1].set_xlabel('Sample')
        axes[1, 1].set_ylabel('Disk %')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()