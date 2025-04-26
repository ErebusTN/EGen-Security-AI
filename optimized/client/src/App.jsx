import React, { useEffect, useState } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { useSelector, useDispatch } from 'react-redux';
import { CssBaseline, CircularProgress, Box } from '@mui/material';

// Import components
import Layout from './components/Layout';
import ProtectedRoute from './components/ProtectedRoute';

// Import pages
import HomePage from './pages/HomePage';
import LoginPage from './pages/LoginPage';
import DashboardPage from './pages/DashboardPage';
import CoursesPage from './pages/CoursesPage';
import CourseDetailPage from './pages/CourseDetailPage';
import SecurityScannerPage from './pages/SecurityScannerPage';
import ProfilePage from './pages/ProfilePage';
import NotFoundPage from './pages/NotFoundPage';
import SettingsPage from './pages/SettingsPage';

// Import authentication actions
import { checkAuth } from './store/slices/authSlice';

const App = () => {
  const dispatch = useDispatch();
  const [isInitialized, setIsInitialized] = useState(false);
  const { isAuthenticated, isLoading } = useSelector((state) => state.auth);

  // Check authentication status on app load
  useEffect(() => {
    const initializeApp = async () => {
      await dispatch(checkAuth());
      setIsInitialized(true);
    };

    initializeApp();
  }, [dispatch]);

  // Show loading spinner while initializing
  if (!isInitialized || isLoading) {
    return (
      <Box 
        sx={{ 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center', 
          height: '100vh' 
        }}
      >
        <CircularProgress />
      </Box>
    );
  }

  return (
    <>
      <CssBaseline />
      <Routes>
        {/* Public routes */}
        <Route path="/" element={<Layout />}>
          <Route index element={<HomePage />} />
          <Route path="login" element={
            isAuthenticated 
              ? <Navigate to="/dashboard" replace /> 
              : <LoginPage />
          } />

          {/* Protected routes */}
          <Route element={<ProtectedRoute isAuthenticated={isAuthenticated} />}>
            <Route path="dashboard" element={<DashboardPage />} />
            <Route path="courses" element={<CoursesPage />} />
            <Route path="courses/:category/:courseId" element={<CourseDetailPage />} />
            <Route path="security-scanner" element={<SecurityScannerPage />} />
            <Route path="profile" element={<ProfilePage />} />
            <Route path="settings" element={<SettingsPage />} />
          </Route>

          {/* 404 route */}
          <Route path="*" element={<NotFoundPage />} />
        </Route>
      </Routes>
    </>
  );
};

export default App; 