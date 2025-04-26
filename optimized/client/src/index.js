import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { Provider } from 'react-redux';
import { ThemeProvider } from '@mui/material/styles';

import App from './App';
import store from './store';
import theme from './theme';
import './index.css';

// Initialize analytics (placeholder)
const initAnalytics = () => {
  console.log('Analytics initialized');
};

// Initialize error tracking (placeholder)
const initErrorTracking = () => {
  console.log('Error tracking initialized');
};

// Initialize app
const init = () => {
  // Only initialize tracking in production
  if (process.env.NODE_ENV === 'production') {
    initAnalytics();
    initErrorTracking();
  }
};

init();

const root = ReactDOM.createRoot(document.getElementById('root'));

root.render(
  <React.StrictMode>
    <Provider store={store}>
      <BrowserRouter>
        <ThemeProvider theme={theme}>
          <App />
        </ThemeProvider>
      </BrowserRouter>
    </Provider>
  </React.StrictMode>
); 