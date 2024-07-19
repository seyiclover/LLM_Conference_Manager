import React, { useState } from 'react';
import styled from 'styled-components';
import LeftSidebar from './LeftSidebar';
import InputArea from '../InputArea/InputArea';
import ChatbotHeader from './ChatbotHeader';
import RightSidebar from '../RightSidebar/RightSidebar';
import ChatArea from '../Message/ChatArea';

const ChatbotContainer = styled.div`
  display: flex;
  height: 100vh;
  background-color: #ffffff;
`;

const MainContent = styled.div`
  flex: 1;
  padding: 20px 0px;
  display: flex;
  flex-direction: column;
  gap: 20px;
  margin-left: 190px;
  margin-right: 400px;
  ::-webkit-scrollbar {
    width: 0px;
  }
`;

const InputAreaWrapper = styled.div`
  position: fixed;
  bottom: 20px;
  left: 220px;
  right: 400px;
  width: calc(100% - 660px);
`;

const Chatbot = () => {
  const [messages, setMessages] = useState([]);

  const handleNewMessage = (message) => {
    console.log('New message:', message); // 새로운 메시지 콘솔 출력
    setMessages((prevMessages) => [...prevMessages, message]);
  };

  return (
    <ChatbotContainer>
      <LeftSidebar />
      <MainContent>
        <ChatbotHeader />
        <ChatArea messages={messages} />
        <InputAreaWrapper>
          <InputArea onMessageSent={handleNewMessage} />
        </InputAreaWrapper>
      </MainContent>
      <RightSidebar />
    </ChatbotContainer>
  );
};

export default Chatbot;