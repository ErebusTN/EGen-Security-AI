import React, { useState } from 'react';
import { Outlet } from 'react-router-dom';
import { useSelector } from 'react-redux';
import { 
  Box, 
  CssBaseline,
  Drawer,
  AppBar,
  Toolbar,
  List,
  Typography,
  Divider,
  IconButton,
  Container,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  useTheme
} from '@mui/material';
import {
  Menu as MenuIcon,
  ChevronLeft as ChevronLeftIcon,
  Dashboard as DashboardIcon,
  School as SchoolIcon,
  Security as SecurityIcon,
  Person as PersonIcon,
  Settings as SettingsIcon,
  Logout as LogoutIcon,
  Home as HomeIcon
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import { useNavigate, useLocation } from 'react-router-dom';

// Import components
import UserMenu from '../UserMenu';
import ThemeToggle from '../ThemeToggle';
import Footer from '../Footer';

const drawerWidth = 240;

const Main = styled('main', { shouldForwardProp: (prop) => prop !== 'open' })(
  ({ theme, open }) => ({
    flexGrow: 1,
    padding: theme.spacing(3),
    transition: theme.transitions.create('margin', {
      easing: theme.transitions.easing.sharp,
      duration: theme.transitions.duration.leavingScreen,
    }),
    marginLeft: `-${drawerWidth}px`,
    ...(open && {
      transition: theme.transitions.create('margin', {
        easing: theme.transitions.easing.easeOut,
        duration: theme.transitions.duration.enteringScreen,
      }),
      marginLeft: 0,
    }),
  }),
);

const AppBarStyled = styled(AppBar, { shouldForwardProp: (prop) => prop !== 'open' })(
  ({ theme, open }) => ({
    transition: theme.transitions.create(['margin', 'width'], {
      easing: theme.transitions.easing.sharp,
      duration: theme.transitions.duration.leavingScreen,
    }),
    ...(open && {
      width: `calc(100% - ${drawerWidth}px)`,
      marginLeft: `${drawerWidth}px`,
      transition: theme.transitions.create(['margin', 'width'], {
        easing: theme.transitions.easing.easeOut,
        duration: theme.transitions.duration.enteringScreen,
      }),
    }),
  }),
);

const DrawerHeader = styled('div')(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  padding: theme.spacing(0, 1),
  // necessary for content to be below app bar
  ...theme.mixins.toolbar,
  justifyContent: 'flex-end',
}));

const Layout = () => {
  const theme = useTheme();
  const navigate = useNavigate();
  const location = useLocation();
  const [open, setOpen] = useState(false);
  const { isAuthenticated, user } = useSelector((state) => state.auth);

  const handleDrawerOpen = () => {
    setOpen(true);
  };

  const handleDrawerClose = () => {
    setOpen(false);
  };

  const mainMenuItems = [
    { 
      text: 'Home', 
      icon: <HomeIcon />, 
      path: '/',
      auth: false  // public route
    },
    { 
      text: 'Dashboard', 
      icon: <DashboardIcon />, 
      path: '/dashboard',
      auth: true  // protected route
    },
    { 
      text: 'Courses', 
      icon: <SchoolIcon />, 
      path: '/courses',
      auth: true  // protected route
    },
    { 
      text: 'Security Scanner', 
      icon: <SecurityIcon />, 
      path: '/security-scanner',
      auth: true  // protected route
    }
  ];

  const secondaryMenuItems = [
    { 
      text: 'Profile', 
      icon: <PersonIcon />, 
      path: '/profile',
      auth: true  // protected route
    },
    { 
      text: 'Settings', 
      icon: <SettingsIcon />, 
      path: '/settings',
      auth: true  // protected route
    }
  ];

  return (
    <Box sx={{ display: 'flex' }}>
      <CssBaseline />
      <AppBarStyled position="fixed" open={open}>
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            onClick={handleDrawerOpen}
            edge="start"
            sx={{ mr: 2, ...(open && { display: 'none' }) }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            EGen Security AI
          </Typography>
          <ThemeToggle />
          {isAuthenticated && <UserMenu user={user} />}
        </Toolbar>
      </AppBarStyled>
      <Drawer
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: drawerWidth,
            boxSizing: 'border-box',
          },
        }}
        variant="persistent"
        anchor="left"
        open={open}
      >
        <DrawerHeader>
          <Typography variant="h6" sx={{ flexGrow: 1, ml: 2 }}>
            Menu
          </Typography>
          <IconButton onClick={handleDrawerClose}>
            <ChevronLeftIcon />
          </IconButton>
        </DrawerHeader>
        <Divider />
        <List>
          {mainMenuItems.map((item) => (
            // Only show auth-required items if authenticated
            (!item.auth || (item.auth && isAuthenticated)) && (
              <ListItem key={item.text} disablePadding>
                <ListItemButton 
                  selected={location.pathname === item.path}
                  onClick={() => navigate(item.path)}
                >
                  <ListItemIcon>
                    {item.icon}
                  </ListItemIcon>
                  <ListItemText primary={item.text} />
                </ListItemButton>
              </ListItem>
            )
          ))}
        </List>
        <Divider />
        <List>
          {isAuthenticated && secondaryMenuItems.map((item) => (
            <ListItem key={item.text} disablePadding>
              <ListItemButton 
                selected={location.pathname === item.path}
                onClick={() => navigate(item.path)}
              >
                <ListItemIcon>
                  {item.icon}
                </ListItemIcon>
                <ListItemText primary={item.text} />
              </ListItemButton>
            </ListItem>
          ))}
          {isAuthenticated && (
            <ListItem disablePadding>
              <ListItemButton onClick={() => navigate('/logout')}>
                <ListItemIcon>
                  <LogoutIcon />
                </ListItemIcon>
                <ListItemText primary="Logout" />
              </ListItemButton>
            </ListItem>
          )}
          {!isAuthenticated && (
            <ListItem disablePadding>
              <ListItemButton onClick={() => navigate('/login')}>
                <ListItemIcon>
                  <PersonIcon />
                </ListItemIcon>
                <ListItemText primary="Login" />
              </ListItemButton>
            </ListItem>
          )}
        </List>
      </Drawer>
      <Main open={open}>
        <DrawerHeader />
        <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
          <Outlet />
        </Container>
        <Footer />
      </Main>
    </Box>
  );
};

export default Layout; 