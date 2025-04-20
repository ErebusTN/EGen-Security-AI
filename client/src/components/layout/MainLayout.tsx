import React, { useState } from 'react';
import { Outlet } from 'react-router-dom';
import {
  Box,
  Toolbar,
  AppBar,
  Drawer,
  IconButton,
  Typography,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Avatar,
  Menu,
  MenuItem,
  Divider,
  useTheme,
  useMediaQuery,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  ModelTraining as TrainingIcon,
  Security as SecurityIcon,
  Insights as MonitoringIcon,
  Settings as SettingsIcon,
  Logout as LogoutIcon,
  Person as PersonIcon,
  Brightness4 as DarkModeIcon,
  Brightness7 as LightModeIcon,
  ChevronLeft as ChevronLeftIcon,
  School as SchoolIcon,
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';
import { useDispatch, useSelector } from 'react-redux';

import { useAuth } from '../../hooks/useAuth';
import { toggleTheme } from '../../context/slices/settingsSlice';
import { RootState } from '../../context/store';

const drawerWidth = 240;

interface ListItemLinkProps {
  to: string;
  primary: string;
  icon?: React.ReactElement;
  isSelected?: boolean;
  onClick?: () => void;
}

const ListItemLink: React.FC<ListItemLinkProps> = ({
  to,
  primary,
  icon,
  isSelected,
  onClick,
}) => {
  const navigate = useNavigate();

  const handleClick = () => {
    navigate(to);
    if (onClick) onClick();
  };

  return (
    <ListItem
      button
      onClick={handleClick}
      selected={isSelected}
      sx={{
        borderRadius: '8px',
        my: 0.5,
        mx: 1,
        '&.Mui-selected': {
          backgroundColor: 'primary.main',
          color: 'white',
          '& .MuiListItemIcon-root': {
            color: 'white',
          },
        },
      }}
    >
      {icon && <ListItemIcon>{icon}</ListItemIcon>}
      <ListItemText primary={primary} />
    </ListItem>
  );
};

const MainLayout: React.FC = () => {
  const theme = useTheme();
  const isSmallScreen = useMediaQuery(theme.breakpoints.down('md'));
  const [open, setOpen] = useState(!isSmallScreen);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const { logout, user } = useAuth();
  const location = useLocation();
  const dispatch = useDispatch();
  const { themeMode } = useSelector((state: RootState) => state.settings);

  const handleDrawerToggle = () => {
    setOpen(!open);
  };

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleLogout = () => {
    handleMenuClose();
    logout();
  };

  const handleThemeToggle = () => {
    dispatch(toggleTheme());
    handleMenuClose();
  };

  const navItems = [
    { path: '/', label: 'Dashboard', icon: <DashboardIcon /> },
    { path: '/training', label: 'Training', icon: <TrainingIcon /> },
    { path: '/courses', label: 'Courses', icon: <SchoolIcon /> },
    { path: '/monitoring', label: 'Monitoring', icon: <MonitoringIcon /> },
    { path: '/security', label: 'Security', icon: <SecurityIcon /> },
    { path: '/settings', label: 'Settings', icon: <SettingsIcon /> },
  ];

  return (
    <Box sx={{ display: 'flex', height: '100vh' }}>
      {/* App Bar */}
      <AppBar
        position="fixed"
        sx={{
          zIndex: (theme) => theme.zIndex.drawer + 1,
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.05)',
          backgroundColor: theme.palette.mode === 'dark' ? '#1f1f1f' : '#ffffff',
          color: theme.palette.mode === 'dark' ? '#ffffff' : '#333333',
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="toggle drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2 }}
          >
            {open ? <ChevronLeftIcon /> : <MenuIcon />}
          </IconButton>
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            EGen Security AI
          </Typography>
          <IconButton
            onClick={handleThemeToggle}
            color="inherit"
            sx={{ ml: 1 }}
          >
            {theme.palette.mode === 'dark' ? <LightModeIcon /> : <DarkModeIcon />}
          </IconButton>
          <IconButton
            onClick={handleMenuOpen}
            size="small"
            sx={{ ml: 2 }}
            aria-controls="menu-appbar"
            aria-haspopup="true"
          >
            <Avatar sx={{ width: 34, height: 34 }}>
              {user?.username?.charAt(0).toUpperCase() || 'U'}
            </Avatar>
          </IconButton>
          <Menu
            id="menu-appbar"
            anchorEl={anchorEl}
            anchorOrigin={{
              vertical: 'bottom',
              horizontal: 'right',
            }}
            keepMounted
            transformOrigin={{
              vertical: 'top',
              horizontal: 'right',
            }}
            open={Boolean(anchorEl)}
            onClose={handleMenuClose}
          >
            <MenuItem onClick={handleMenuClose}>
              <ListItemIcon>
                <PersonIcon fontSize="small" />
              </ListItemIcon>
              <ListItemText primary="Profile" />
            </MenuItem>
            <MenuItem onClick={handleThemeToggle}>
              <ListItemIcon>
                {themeMode === 'dark' ? <LightModeIcon fontSize="small" /> : <DarkModeIcon fontSize="small" />}
              </ListItemIcon>
              <ListItemText primary={`${themeMode === 'dark' ? 'Light' : 'Dark'} Mode`} />
            </MenuItem>
            <Divider />
            <MenuItem onClick={handleLogout}>
              <ListItemIcon>
                <LogoutIcon fontSize="small" color="error" />
              </ListItemIcon>
              <ListItemText primary="Logout" sx={{ color: 'error.main' }} />
            </MenuItem>
          </Menu>
        </Toolbar>
      </AppBar>

      {/* Sidebar */}
      <Drawer
        variant={isSmallScreen ? 'temporary' : 'persistent'}
        open={isSmallScreen ? open : true}
        onClose={isSmallScreen ? handleDrawerToggle : undefined}
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: drawerWidth,
            boxSizing: 'border-box',
            backgroundColor: theme.palette.mode === 'dark' ? '#262626' : '#f8f8f8',
            border: 'none',
            boxShadow: theme.palette.mode === 'dark' ? 'none' : '4px 0 8px rgba(0, 0, 0, 0.05)',
          },
        }}
      >
        <Toolbar />
        <Box sx={{ overflow: 'auto', px: 2, py: 2 }}>
          <List>
            {navItems.map((item) => (
              <ListItemLink
                key={item.path}
                to={item.path}
                primary={item.label}
                icon={item.icon}
                isSelected={location.pathname === item.path}
                onClick={isSmallScreen ? handleDrawerToggle : undefined}
              />
            ))}
          </List>
          <Divider sx={{ my: 2 }} />
          <Box
            sx={{
              p: 2,
              backgroundColor: 'background.paper',
              borderRadius: '8px',
              textAlign: 'center',
            }}
          >
            <Typography variant="body2" color="text.secondary">
              Model: Security v1.0
            </Typography>
            <Typography variant="caption" display="block" color="success.main" sx={{ mt: 1 }}>
              Status: Online
            </Typography>
          </Box>
        </Box>
      </Drawer>

      {/* Main content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          width: { md: `calc(100% - ${drawerWidth}px)` },
          overflow: 'auto',
        }}
      >
        <Toolbar />
        <Outlet />
      </Box>
    </Box>
  );
};

export default MainLayout; 