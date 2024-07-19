// src/App.js
import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import { UserProvider } from './contexts/UserContext';
import MainPage from './pages/MainPage';
import LoginPage from './pages/LoginPage';
import ChatbotComponent from './components/Chatbot/Chatbot';
import ProtectedRoute from './contexts/ProtectedRoute';

function App() {
  return (
    <Router>
      <UserProvider>
          <Routes>
            <Route path="/" element={<MainPage />} />
            <Route path="/login" element={<LoginPage />} />
            <Route
              path="/chatbot"
              element={
                <ProtectedRoute>
                  <ChatbotComponent />
                </ProtectedRoute>
              }
            />
          </Routes>
        
      </UserProvider>
    </Router>
  );
}

export default App;
