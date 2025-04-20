import React, { useState, useEffect } from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  CircularProgress,
  Box,
  Stack,
  Chip,
  LinearProgress,
  Alert,
} from '@mui/material';
import {
  Security as SecurityIcon,
  Speed as SpeedIcon,
  Memory as MemoryIcon,
  Storage as StorageIcon,
  BugReport as BugIcon,
  CheckCircle as CheckCircleIcon,
} from '@mui/icons-material';

// Mock data - In a real app, this would come from API
const mockModelStats = {
  name: 'Security AI v1.0',
  status: 'Online',
  lastUpdated: '2023-04-18 14:30:22',
  threatDetectionRate: 93.8,
  falsePositiveRate: 2.4,
  averageResponseTime: 124, // ms
  gpuUtilization: 42, // percent
  ramUtilization: 72, // percent
  activeRequests: 6,
  totalRequestsToday: 1247,
  recentThreats: [
    { id: 1, type: 'Malware', severity: 'High', timestamp: '14:22:15', status: 'Mitigated' },
    { id: 2, type: 'Phishing', severity: 'Medium', timestamp: '14:15:33', status: 'Active' },
    { id: 3, type: 'DDoS', severity: 'High', timestamp: '13:58:02', status: 'Investigating' },
    { id: 4, type: 'SQLi', severity: 'Medium', timestamp: '13:45:21', status: 'Mitigated' },
  ],
};

// Severity to color mapping
const severityColors = {
  High: 'error',
  Medium: 'warning',
  Low: 'info',
};

const StatusChip = ({ status }) => {
  const getColor = () => {
    switch (status) {
      case 'Online':
        return 'success';
      case 'Offline':
        return 'error';
      case 'Maintenance':
        return 'warning';
      default:
        return 'default';
    }
  };

  return (
    <Chip
      label={status}
      color={getColor()}
      size="small"
      variant="filled"
      sx={{ fontWeight: 'bold' }}
    />
  );
};

const ThreatItem = ({ threat }) => {
  const getStatusColor = () => {
    switch (threat.status) {
      case 'Mitigated':
        return 'success';
      case 'Active':
        return 'error';
      case 'Investigating':
        return 'warning';
      default:
        return 'default';
    }
  };

  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        p: 1,
        mb: 1,
        borderRadius: 1,
        backgroundColor: (theme) =>
          theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.03)',
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center' }}>
        <BugIcon color="error" sx={{ mr: 1 }} />
        <Box>
          <Typography variant="body2" fontWeight="medium">
            {threat.type}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {threat.timestamp}
          </Typography>
        </Box>
      </Box>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <Chip
          label={threat.severity}
          color={severityColors[threat.severity] || 'default'}
          size="small"
          variant="outlined"
        />
        <Chip
          label={threat.status}
          color={getStatusColor()}
          size="small"
          variant="filled"
          sx={{ fontWeight: 'bold' }}
        />
      </Box>
    </Box>
  );
};

const StatCard = ({ title, value, icon, color, suffix = '', helperText = '' }) => {
  return (
    <Card sx={{ height: '100%', position: 'relative', overflow: 'visible' }}>
      <Box
        sx={{
          position: 'absolute',
          top: -20,
          left: 20,
          width: 56,
          height: 56,
          borderRadius: '12px',
          backgroundColor: `${color}.main`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
        }}
      >
        {icon}
      </Box>
      <CardContent sx={{ pt: 4, mt: 2 }}>
        <Typography variant="h5" fontWeight="bold" mt={1}>
          {value}
          {suffix && <Typography component="span" variant="body1" ml={0.5}>{suffix}</Typography>}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          {title}
        </Typography>
        {helperText && (
          <Typography variant="caption" color="text.secondary" display="block" mt={1}>
            {helperText}
          </Typography>
        )}
      </CardContent>
    </Card>
  );
};

const Dashboard: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState(null);

  // Simulate API fetch
  useEffect(() => {
    const fetchStats = async () => {
      // In a real app, this would be an API call
      await new Promise((resolve) => setTimeout(resolve, 1500));
      setStats(mockModelStats);
      setLoading(false);
    };

    fetchStats();

    // Set up interval to refresh data (simulating real-time updates)
    const intervalId = setInterval(() => {
      const updatedStats = { ...mockModelStats };
      // Simulate changing values
      updatedStats.activeRequests = Math.floor(Math.random() * 10) + 2;
      updatedStats.gpuUtilization = Math.floor(Math.random() * 30) + 30;
      updatedStats.ramUtilization = Math.floor(Math.random() * 20) + 60;
      setStats(updatedStats);
    }, 5000);

    return () => clearInterval(intervalId);
  }, []);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '70vh' }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Grid container spacing={3}>
      {/* Header */}
      <Grid item xs={12}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h4" fontWeight="bold">
            Security Dashboard
          </Typography>
          <Box>
            <StatusChip status={stats.status} />
          </Box>
        </Box>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          Model: {stats.name} | Last Updated: {stats.lastUpdated}
        </Typography>
      </Grid>

      {/* Key Metrics */}
      <Grid item xs={12} md={6} lg={3}>
        <StatCard
          title="Threat Detection Rate"
          value={stats.threatDetectionRate}
          suffix="%"
          icon={<SecurityIcon sx={{ color: 'white', fontSize: 28 }} />}
          color="success"
          helperText="Overall accuracy in threat detection"
        />
      </Grid>
      <Grid item xs={12} md={6} lg={3}>
        <StatCard
          title="False Positive Rate"
          value={stats.falsePositiveRate}
          suffix="%"
          icon={<CheckCircleIcon sx={{ color: 'white', fontSize: 28 }} />}
          color="error"
          helperText="Rate of false positive detections"
        />
      </Grid>
      <Grid item xs={12} md={6} lg={3}>
        <StatCard
          title="Response Time"
          value={stats.averageResponseTime}
          suffix="ms"
          icon={<SpeedIcon sx={{ color: 'white', fontSize: 28 }} />}
          color="info"
          helperText="Average model response time"
        />
      </Grid>
      <Grid item xs={12} md={6} lg={3}>
        <StatCard
          title="Requests Today"
          value={stats.totalRequestsToday}
          icon={<StorageIcon sx={{ color: 'white', fontSize: 28 }} />}
          color="warning"
          helperText={`${stats.activeRequests} active requests`}
        />
      </Grid>

      {/* Resource Utilization */}
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Resource Utilization
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Box sx={{ mb: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2">GPU Utilization</Typography>
                    <Typography variant="body2">{stats.gpuUtilization}%</Typography>
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={stats.gpuUtilization}
                    color={stats.gpuUtilization > 80 ? 'error' : 'primary'}
                    sx={{
                      height: 8,
                      borderRadius: 2,
                    }}
                  />
                </Box>
              </Grid>
              <Grid item xs={12} md={6}>
                <Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2">RAM Utilization</Typography>
                    <Typography variant="body2">{stats.ramUtilization}%</Typography>
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={stats.ramUtilization}
                    color={stats.ramUtilization > 80 ? 'error' : 'primary'}
                    sx={{
                      height: 8,
                      borderRadius: 2,
                    }}
                  />
                </Box>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      </Grid>

      {/* Recent Threats */}
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Recent Threats
            </Typography>
            {stats.recentThreats.length > 0 ? (
              <Stack spacing={1}>
                {stats.recentThreats.map((threat) => (
                  <ThreatItem key={threat.id} threat={threat} />
                ))}
              </Stack>
            ) : (
              <Alert severity="info">No recent threats detected.</Alert>
            )}
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

export default Dashboard;
