import React, { useEffect, useCallback } from 'react';
import { GoogleLogin } from '@react-oauth/google';
import { GoogleOAuthProvider } from '@react-oauth/google';
import { useUser } from '../contexts/UserContext';
import {jwtDecode} from 'jwt-decode';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { LoginPageContainer, LoginTitle } from './LoginPage.styled';
import Nexologo from '../images/Nexo_logo_small.png';


function LoginPage() {
  const { setUser } = useUser();
  const navigate = useNavigate();

   // 인증 상태를 확인하는 함수
  const checkAuthStatus = useCallback(async () => {
    try {
      const token = localStorage.getItem('access_token');
      if (!token) {
        throw new Error('No token found');
      }

      const response = await axios.get('http://localhost:8000/auth/secure-data', {
        headers: {
          Authorization: `Bearer ${token}`, // 요청 헤더에 토큰 포함
        },
      });

      const user = response.data.user;
      if (user) {
        setUser({
          name: user.name,
          email: user.email,
          imageUrl: user.image,
        });
        navigate('/chatbot'); // 인증 성공 시 챗봇 페이지로 이동
      } else {
        console.error('User data is missing in response');
      }
    } catch (error) {
      console.error('Not authenticated:', error.response ? error.response.data : error.message);
    }
  }, [setUser, navigate]);

  useEffect(() => {
    checkAuthStatus();
  }, [checkAuthStatus]);

  // 로그인 성공 시 호출되는 함수
  const onSuccess = async (response) => {
    const googleToken = response.credential;
    const decoded = jwtDecode(googleToken);
    console.log('Decoded Token:', decoded);

    try {
      const result = await axios.post('http://localhost:8000/auth/callback', {
        token: googleToken,
      }, {
        headers: {
          'Content-Type': 'application/json',
        },
      });

      const accessToken = result.data.token;
      console.log('Access Token:', accessToken);  // 디버깅용으로 토큰 출력
      localStorage.setItem('access_token', accessToken); // local storage에 토큰 저장

      // 토큰 디코딩 후 사용자 상태 설정
      const decodedAccessToken = jwtDecode(accessToken);

      setUser({
        name: decodedAccessToken.name,
        email: decodedAccessToken.email,
        imageUrl: decodedAccessToken.picture,  // JWT 토큰의 'picture' 필드를 사용
      });

      navigate('/chatbot'); // 로그인 성공 시 챗봇 페이지로 이동
    } catch (error) {
      console.error('Login Failed:', error.response ? error.response.data : error.message);
    }
  };
  // 로그인 실패 시 호출되는 함수
  const onFailure = (response) => {
    console.error('Login Failed:', response);
  };
  

  return (
    <GoogleOAuthProvider clientId={process.env.REACT_APP_GOOGLE_CLIENT_ID}>
      <LoginPageContainer>
        <img src={Nexologo} alt="NexoLogo" className="logo" />
        <LoginTitle>Login with Google</LoginTitle>
        <GoogleLogin onSuccess={onSuccess} onFailure={onFailure} />
      </LoginPageContainer>
    </GoogleOAuthProvider>
  );
}

export default LoginPage;