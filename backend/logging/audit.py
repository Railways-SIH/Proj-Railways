# backend/logging/audit.py

from datetime import datetime, timedelta
from typing import Dict, List, Any
class AuditLogger:
    def __init__(self):
        self.audit_log = []
        self.kpi_history = []
        
    def log_recommendation(self, recommendation, accepted=False):
        """Log optimization recommendations"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'recommendation',
            'data': recommendation,
            'accepted': accepted
        }
        self.audit_log.append(log_entry)
        
    def log_kpi(self, kpis):
        """Log KPI snapshots"""
        kpi_entry = {
            'timestamp': datetime.now().isoformat(),
            'kpis': kpis
        }
        self.kpi_history.append(kpi_entry)
        
    def get_recent_logs(self, limit=50):
        """Get recent audit logs"""
        return self.audit_log[-limit:]
        
    def get_kpi_history(self, hours=24):
        """Get KPI history for specified hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [entry for entry in self.kpi_history 
                if datetime.fromisoformat(entry['timestamp']) > cutoff]

