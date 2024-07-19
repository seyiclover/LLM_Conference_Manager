import React, { useRef, useEffect } from 'react';
import styled from 'styled-components';
import userPlaceholder from '../../images/user.png';
import nexochatSmallImage from '../../images/nexochat_small.png';
import { useUser } from '../../contexts/UserContext';
import ReactMarkdown from 'react-markdown';

const MessagesContainer = styled.div`
  flex: 1;
  overflow-y: auto;
  padding-left: 20px;
  display: flex;
  flex-direction: column;
  gap: 5px;
  margin-bottom: 80px;
`;

const Message = styled.div`
  display: flex;
  align-items: flex-start;
  padding: 8px;
  border-radius: 10px;
  gap: 10px;
`;

const MessageContent = styled.div`
  display: flex;
  flex-direction: column;
  justify-content: center; /* 추가: 내용이 중앙에 오도록 조정 */
  gap: 2px;
  max-width: 88%;  /* 최대 너비 설정 */
  word-break: break-word;  /* 단어가 길 경우 줄바꿈 */
`;

const Sender = styled.div`
  font-weight: bold;
  margin-bottom: 1px;
`;

const ImageContainer = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  width: 50px;  // 모든 이미지를 위한 균일한 너비 설정
`;

const StyledImage = styled.img`
  width: ${({ isUser }) => (isUser ? '40px' : '50px')};
  height: ${({ isUser }) => (isUser ? '40px' : '50px')};
  border-radius: 50%;
`;
// 마크다운 요소들에 대한 스타일 정의
const  MarkdownWrapper = styled.div`
 & > * {
    margin-top: 3px;
  }
`;
const ChatArea = ({ messages }) => {
  const { user } = useUser();
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <MessagesContainer>
      {messages.map((msg, index) => (
        <Message key={index} role={msg.role}>
          <ImageContainer>
            <StyledImage
              src={msg.role === 'user' ? (user && user.imageUrl ? user.imageUrl : userPlaceholder) : nexochatSmallImage}
              alt={msg.role}
              isUser={msg.role === 'user'}
            />
          </ImageContainer>
          <MessageContent>
            <Sender>{msg.role === 'user' ? (user && user.name ? user.name : 'User') : 'Nexochat'}</Sender>
            <MarkdownWrapper>
            <ReactMarkdown>{msg.content}</ReactMarkdown>
            </MarkdownWrapper>
          </MessageContent>
        </Message>
      ))}
      <div ref={messagesEndRef} />
    </MessagesContainer>
  );
};

export default ChatArea;
