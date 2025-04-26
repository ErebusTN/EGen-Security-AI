import { configureStore } from '@reduxjs/toolkit';
import { setupListeners } from '@reduxjs/toolkit/query';

// Import reducers
import authReducer from './slices/authSlice';
import uiReducer from './slices/uiSlice';
import courseReducer from './slices/courseSlice';
import securityReducer from './slices/securitySlice';

// Import API services
import { api } from './api';

// Configure Redux store
const store = configureStore({
  reducer: {
    auth: authReducer,
    ui: uiReducer,
    courses: courseReducer,
    security: securityReducer,
    [api.reducerPath]: api.reducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware().concat(api.middleware),
  devTools: process.env.NODE_ENV !== 'production',
});

// Setup listeners for RTK Query
setupListeners(store.dispatch);

export default store; 