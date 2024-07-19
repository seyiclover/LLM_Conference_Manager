import React from 'react';
import styled from 'styled-components';
import nexochatImage from '../../images/nexochat.png';
import userPlaceholder from '../../images/user.png';
import { useUser } from '../../contexts/UserContext';
import { useNavigate } from 'react-router-dom';

// 스타일 정의
const SidebarContainer = styled.div`
  width: 175px;
  background-color: #3788B4;
  left: 10px;
  top: 15px;
  bottom: 15px;
  display: flex;
  flex-direction: column;
  align-items: center;
  color: white;
  border-radius: 30px 10px 10px 30px;
  position:fixed;
`;

const UserProfile = styled.div`
  display: flex;
  align-items: center;
  flex-direction: column;
  margin-top: auto;
  margin-bottom: 50px;
`;

const UserProfileImage = styled.img`
  width: 50px;
  height: 50px;
  border-radius: 50%;
  margin-bottom: 10px;
`;

const NexoLogo = styled.img`
    width: 135px;
    height: 104px;
    margin-bottom: 20px;
    margin-top: 50px;
`;
const LeftSidebar = () => {
  const navigate = useNavigate(); // 페이지 이동을 위한 훅
  const { user } = useUser(); // 사용자 정보 가져오기


  // 로고 클릭 시 메인 페이지로 이동하는 함수
  const handleLogoClick = () => {
    navigate('/');
  };

  return (
    <SidebarContainer>
        {/* NexoChat 로고 클릭 시 메인 페이지로 이동 */}
        <NexoLogo src={nexochatImage} alt="NexoChat Logo" onClick={handleLogoClick} style={{ cursor: 'pointer' }} />
        {/* 사용자 프로필 표시 */}
        <UserProfile>
          <UserProfileImage src={user ? user.imageUrl : userPlaceholder} alt="User" />
          <span>{user ? user.name : 'USER'}</span>
        </UserProfile>

    </SidebarContainer>
  );
};

// 컴포넌트를 기본으로 익스포트
export default LeftSidebar;


