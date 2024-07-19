import React from 'react';
import { Navigate } from 'react-router-dom';
import { useUser } from './UserContext';

const ProtectedRoute = ({ children }) => {
  const { user } = useUser(); // 현재 사용자 상태를 가져옴

  if (!user) {
    // 사용자가 인증되지 않은 경우, 로그인 페이지로 리디렉션
    return <Navigate to="/login" replace />;
  }

  // 사용자가 인증된 경우, 자식 컴포넌트 렌더링
  return children;
};

export default ProtectedRoute;