"""
Server Manager for CoT Analysis

Manages spawning and lifecycle of separate CoT analysis server instances.
"""

import subprocess
import time
import json
import os
from pathlib import Path
from typing import Dict, Optional, List
import logging
import requests
from datetime import datetime

logger = logging.getLogger(__name__)

class CoTServerManager:
    """Manages separate CoT analysis server instances"""
    
    def __init__(self, base_port: int = 9000, server_script: Optional[Path] = None):
        """
        Initialize the server manager.
        
        Args:
            base_port: Starting port number for analysis servers
            server_script: Path to the CoT analysis server script
        """
        self.base_port = base_port
        self.current_port = base_port
        self.running_servers: Dict[str, Dict] = {}  # cot_job_id -> server info
        self.port_usage: Dict[int, str] = {}  # port -> cot_job_id
        
        # Find server script
        if server_script is None:
            script_dir = Path(__file__).parent
            server_script = script_dir / "cot_analysis_server.py"
        self.server_script = server_script
        
        # Create logs directory for server processes
        self.logs_dir = Path(__file__).parent.parent / "logs" / "cot_servers"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
    
    def find_available_port(self) -> int:
        """Find an available port starting from current_port"""
        max_attempts = 100
        port = self.current_port
        
        for _ in range(max_attempts):
            # Check if port is in use
            try:
                result = subprocess.run(
                    ["netstat", "-tuln"],
                    capture_output=True,
                    text=True,
                    timeout=1
                )
                if f":{port} " not in result.stdout:
                    self.current_port = port + 1
                    return port
            except:
                # Fallback: try to connect
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    sock.bind(('127.0.0.1', port))
                    sock.close()
                    self.current_port = port + 1
                    return port
                except:
                    pass
            
            port += 1
        
        raise RuntimeError(f"Could not find available port after {max_attempts} attempts")
    
    def start_server(self, cot_job_id: str, job_id: str, config: Dict) -> Dict:
        """
        Start a new CoT analysis server instance.
        
        Args:
            cot_job_id: The CoT analysis job ID
            job_id: The parent job ID
            config: Configuration dictionary
            
        Returns:
            Dictionary with server info (port, pid, etc.)
        """
        port = self.find_available_port()
        
        # Log file for this server
        log_file = self.logs_dir / f"cot_server_{cot_job_id}_{port}.log"
        
        # Start the server process
        cmd = [
            "python3",
            str(self.server_script),
            "--port", str(port),
            "--host", "127.0.0.1"
        ]
        
        logger.info(f"Starting CoT analysis server for {cot_job_id} on port {port}")
        
        try:
            # Start process in background
            process = subprocess.Popen(
                cmd,
                stdout=open(log_file, 'w'),
                stderr=subprocess.STDOUT,
                cwd=str(self.server_script.parent.parent),
                env=os.environ.copy()
            )
            
            # Wait a bit for server to start
            time.sleep(2)
            
            # Check if process is still running
            if process.poll() is not None:
                # Process died, read log
                with open(log_file, 'r') as f:
                    log_content = f.read()
                raise RuntimeError(f"Server failed to start. Log: {log_content[:500]}")
            
            # Verify server is responding
            max_retries = 10
            for i in range(max_retries):
                try:
                    response = requests.get(f"http://127.0.0.1:{port}/health", timeout=1)
                    if response.status_code == 200:
                        break
                except:
                    if i < max_retries - 1:
                        time.sleep(1)
                    else:
                        raise RuntimeError(f"Server started but not responding on port {port}")
            
            # Trigger the analysis asynchronously (don't wait for completion)
            # The analysis will run in the background and update job_db directly
            try:
                analysis_response = requests.post(
                    f"http://127.0.0.1:{port}/run-analysis",
                    json={
                        "cot_job_id": cot_job_id,
                        "job_id": job_id,
                        "config": config
                    },
                    timeout=2  # Short timeout, just to verify the request was accepted
                )
                # Don't fail if it times out - the analysis runs async in the server
            except requests.exceptions.Timeout:
                # This is expected - the analysis runs asynchronously
                logger.info(f"Analysis request sent to server (running async)")
            except Exception as e:
                logger.warning(f"Could not verify analysis start: {e}, but server is running")
            
            # Store server info
            server_info = {
                "cot_job_id": cot_job_id,
                "job_id": job_id,
                "port": port,
                "pid": process.pid,
                "process": process,
                "log_file": str(log_file),
                "started_at": time.time(),
                "config": config
            }
            
            self.running_servers[cot_job_id] = server_info
            self.port_usage[port] = cot_job_id
            
            logger.info(f"CoT analysis server started successfully: {cot_job_id} on port {port} (PID: {process.pid})")
            
            return server_info
            
        except Exception as e:
            logger.error(f"Failed to start server for {cot_job_id}: {e}")
            # Cleanup on failure
            if cot_job_id in self.running_servers:
                self.stop_server(cot_job_id)
            raise
    
    def stop_server(self, cot_job_id: str) -> bool:
        """
        Stop a running server instance.
        
        Args:
            cot_job_id: The CoT analysis job ID
            
        Returns:
            True if server was stopped, False if not found
        """
        if cot_job_id not in self.running_servers:
            return False
        
        server_info = self.running_servers[cot_job_id]
        process = server_info.get("process")
        port = server_info.get("port")
        
        logger.info(f"Stopping CoT analysis server: {cot_job_id} on port {port}")
        
        try:
            if process and process.poll() is None:
                # Process is still running
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        except Exception as e:
            logger.warning(f"Error stopping process for {cot_job_id}: {e}")
        
        # Cleanup
        del self.running_servers[cot_job_id]
        if port in self.port_usage:
            del self.port_usage[port]
        
        return True
    
    def get_server_info(self, cot_job_id: str) -> Optional[Dict]:
        """Get information about a running server"""
        return self.running_servers.get(cot_job_id)
    
    def list_running_servers(self) -> List[Dict]:
        """List all running servers"""
        return list(self.running_servers.values())
    
    def cleanup_completed_servers(self):
        """Clean up servers for completed jobs"""
        from .runner import job_db
        
        to_remove = []
        for cot_job_id, server_info in self.running_servers.items():
            queue_entry = job_db.get(f"cot_analysis_{cot_job_id}")
            if queue_entry:
                status = queue_entry.get("status")
                if status in ["DONE", "ERROR"]:
                    # Job is complete, stop the server
                    logger.info(f"Cleaning up completed server: {cot_job_id}")
                    self.stop_server(cot_job_id)
                    to_remove.append(cot_job_id)
        
        return len(to_remove)

# Global server manager instance
_server_manager: Optional[CoTServerManager] = None

def get_server_manager() -> CoTServerManager:
    """Get or create the global server manager instance"""
    global _server_manager
    if _server_manager is None:
        _server_manager = CoTServerManager()
    return _server_manager

