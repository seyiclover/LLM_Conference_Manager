import React from 'react';
import styled from 'styled-components';
import searchIcon from '../../images/search_icon.png';
const HeaderContainer = styled.header`
  width: 95%;
  background-color: #fff;
  padding: 10px 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
`;

const SearchInputWrapper = styled.div`
  position: relative;
  left: 245px;
`;

const SearchInput = styled.input`
  width: 248px;
  padding: 10px 40px 10px 40px;
  border: 1.47368px solid rgba(145, 151, 179, 0.5);
  border-radius: 20px;
  box-sizing: border-box;
  background: #ffffff;
  height: 40px;
  &:focus {
    outline: none;
  }
`;

const SearchImg = styled.img`
  width: 20px;
  position: absolute;
  height: 20px;
  top: 10px;
  left: 10px;
`;

const BottomLine = styled.div`
  width: 90%;
  height: 2px;
  background-color: #3788B4;
  position: absolute;
  bottom: -1px;
`;

const ChatbotHeader = () => {
  return (
    <HeaderContainer>
      <SearchInputWrapper>
        <SearchImg src={searchIcon} alt="Search Icon" />
        <SearchInput type="text" placeholder="검색어를 입력하세요" />
      </SearchInputWrapper>
      <BottomLine />
    </HeaderContainer>
  );
};

export default ChatbotHeader;
