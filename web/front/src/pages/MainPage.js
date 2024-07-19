// src/pages/MainPage.js
import React from 'react';
import { useUser } from '../contexts/UserContext';
import { useMediaQuery } from 'react-responsive';
import { useNavigate } from 'react-router-dom';
import styled from 'styled-components';

// 이미지 import
import NexoChat_logo from '../images/Nexo_logo_noname1.png';
import Circle from '../images/circle.png';
import Stt_logo from '../images/stt_logo.png';
import Arrow from '../images/arrow.png';
import Meeting from '../images/meeting_logo.png';
import Robot from '../images/robot.png';
import Clova from '../images/clova.png';
import Notion from '../images/notion.png';
import Openai from '../images/openai.png';

// 스타일 정의
const MainPageContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 100%;
  margin: 0 auto;
  padding: 0;
  font-family: 'Roboto', sans-serif;
`;

const Navbar = styled.div`
  background-color: #f8f9fa;
  display: flex;
  align-items: center;
  position: fixed;
  top: 0;
  width: 100%;
  z-index: 1000;
  height: 5rem;
  justify-content: space-between;
  box-shadow: 0 0.25rem 0.25rem rgba(0, 0, 0, 0.4);
  margin-bottom: 0;

  img {
    margin-left: 4.6875rem;
    width: 5.625rem;
    height: 5.2rem;
    padding-top: 0.625rem;
  }

  a {
    text-decoration: none;
    color: #000;
    margin-right: 1.25rem;

    &:hover {
      text-decoration: underline;
    }
  }
`;

const NavText = styled.span`
  font-weight: 900;
  font-size: 2.5rem;
  color: #05305b;
  position: absolute;
  left: 10rem;
  top: 1rem;
`;

const NavRight = styled.div`
  margin-left: auto;
  display: flex;
  align-items: center;
`;

const LoginLink = styled.div`
  color: #3788b4 !important;
  font-family: 'Roboto';
  font-weight: 700;
  font-size: 1.25rem;
  margin-right: 2.188rem !important;
  cursor: pointer;
`;

const SignUp = styled.div`
  color: #ffffff !important;
  background-color: #3788b4;
  padding: 0.3125rem 4.375rem;
  font-family: 'Roboto';
  font-weight: 700;
  font-size: 1.25rem;
  margin-right: 4.375rem !important;
  cursor: pointer;
`;

const LogoutButton = styled.button`
  color: #ffffff !important;
  background-color: #3788b4;
  padding: 0.3125rem 4.375rem;
  font-family: 'Roboto';
  font-weight: 700;
  font-size: 1.25rem;
  margin-right: 3.125rem;
  border: none;
  cursor: pointer;
`;

const MainSection = styled.div`
  display: flex;
  flex-direction: row;
  align-items: flex-start;
  justify-content: center;
  width: 100%;
  margin: 0 auto;
  padding-top: 3.8rem;
  box-sizing: border-box;
`;

const LeftSection = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  position: relative;
  padding-top: 2rem;
  text-align: center;
  margin-top: 0rem;

  p {
    margin-top: 6.813rem;
    font-weight: 900;
    font-size: 2.1875rem;
    color: #05305b;
    z-index: 2;
    margin-bottom: 0;
    margin-right: 10rem;
  }

  .image-container {
    display: flex;
    flex-direction: row;
    justify-content: center;
    align-items: center;
    margin-right: 10rem;

    img {
      margin-right: 1rem;
      height: auto;
      max-width: 100%;
      z-index: 2;
    }
  }

  img.robot {
    max-width: 26.1875rem;
    width: 100%;
    z-index: 3;
    margin-right: 13rem;
  }

  img.circle {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: auto;
    z-index: 1;
  }
`;

const RightContainer = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-top: 7rem;
`;

const TitleSection = styled.div`
  width: 100%;
  text-align: center;
  h1 {
    font-weight: 900;
    color: #05305b;
    font-family: 'Roboto Condensed';
    font-size: 8rem;
    font-style: normal;
    line-height: 9.125rem;
    height: 7.3rem;
    letter-spacing: -0.1rem;
  }
  p {
    font-weight: 900;
    font-size: 2rem;
    color: #000000;
    margin-top: -5.2rem;
  }
`;

const CTAButton = styled.div`
  background-color: #3788b4;
  color: #ffffff;
  padding: 1.25rem;
  border: none;
  border-radius: 1.25rem;
  font-size: 1.5625rem;
  font-weight: 700;
  cursor: pointer;
  margin-top: 2rem;
  width: 11.625rem;
  text-decoration: none;
  display: inline-block;
  text-align: center;
`;

const Partners = styled.div`
  width: 100%;
  display: flex;
  justify-content: center;
  margin-top: 2rem;
  margin-right: 2rem;

  img {
    width: 12.5rem;
    object-fit: contain;
    height: 13rem;
  }
  img:nth-child(n+2) {
    margin-left: 1.5rem;
  }

  .zoomClova {
    transform: scale(1.5);
  }
  .zoomNotion {
    transform: scale(1.2);
  }
`;

const MainPage = () => {
  const { user } = useUser();
  const isLaptopOrDesktop = useMediaQuery({ query: '(min-width: 1024px)' });
  
  const navigate = useNavigate();


  // 로그아웃
  const handleLogout = () => {
    localStorage.removeItem('access_token');
    window.location.reload();
  };

  // Sign Up 버튼 클릭 핸들러
  const handleSignUp = () => {
    window.location.href = 'https://accounts.google.com/signup';
  };

  // 로그인 버튼 클릭 핸들러
  const handleLogin = () => {
    navigate('/login');
  };

  return (
    <MainPageContainer>
      <Navbar>
        <div>
          <img src={NexoChat_logo} alt="NexoChat Logo" className="navlogo" />
          <NavText>NexoChat</NavText>
        </div>
        <NavRight>
          {user ? (
            <LogoutButton onClick={handleLogout}>Logout</LogoutButton>
          ) : (
            <>
              <LoginLink onClick={handleLogin}>Login</LoginLink>
              <SignUp onClick={handleSignUp}>Sign Up</SignUp> {/* Sign Up 버튼 */}
            </>
          )}
        </NavRight>
      </Navbar>
      {isLaptopOrDesktop && (
        <MainSection>
          <LeftSection>
            <p>Voice-to-Text Conversion</p>
            <img src={Circle} alt="circle" className="circle" />
            <div className="image-container">
              <img src={Stt_logo} alt="stt" className="stt" />
              <img src={Arrow} alt="arrow" className="arrow" />
              <img src={Meeting} alt="meeting" className="meeting" />
            </div>
            <img src={Robot} alt="Robot" className="robot" />
          </LeftSection>
          <RightContainer>
            <TitleSection>
              <h1>NexoChat</h1>
              <p>Streamline Meetings with Voice-to-Text</p>
              <CTAButton onClick={() => navigate('/chatbot')}>GET STARTED</CTAButton>
            </TitleSection>
            <Partners>
              <img src={Clova} alt="Clova Logo" className="zoomClova" />
              <img src={Notion} alt="Notion Logo" className="zoomNotion" />
              <img src={Openai} alt="OpenAI Logo" />
            </Partners>
          </RightContainer>
        </MainSection>
      )}
    </MainPageContainer>
  );
};

export default MainPage;
