import React, { useState } from 'react';
import styled from 'styled-components';
import axios from 'axios';
import inputButton from '../../images/button.png';
import LoadingSpinner from '../../utils/spinner_chat';

const InputContainer = styled.div`
  display: flex;
  align-items: center;
  padding: 10px;
  background-color: #E4DFDF;
  border-radius: 30px;
  position: relative;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  border: 1px solid #3788B4;
  margin-top: 5px;
`;

const TextInput = styled.input`
  flex-grow: 1;
  border: none;
  padding: 6px 5px 5px 10px;
  margin-right: 10px;
  background-color: #E4DFDF;
  font-size: 16px;
  font-family: Montserrat;
  font-weight: medium;
  &:focus {
    outline: none;
  }
`;

const SendButton = styled.button`
  border: none;
  background-color: #E4DFDF;
  padding-top: 3px;
  cursor: pointer;
`;

const InputArea = ({ onMessageSent }) => {
  const [input, setInput] = useState('');
  const [isSending, setIsSending] = useState(false);

  const handleSend = async () => {
    if (input.trim() && !isSending) {
      setIsSending(true);

      const userMessage = {
        role: 'user',
        content: input,
      };

      onMessageSent(userMessage);
      setInput('');

      try {
        const response = await axios.post(
          'http://localhost:8000/clova/chat',
          { question: userMessage.content },
          {
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${localStorage.getItem('access_token')}`
            }
          }
        );
       
        const assistantMessage = {
          role: 'assistant',
          content: response.data.response
        };
        onMessageSent(assistantMessage); // 서버 응답을 화면에 표시
      } catch (error) {
        console.error("Error sending message:", error);
        if (error.response && error.response.status === 403) {
          console.error("Authorization error, please check the access token.");
        }
        if (error.response) {
          switch (error.response.status) {
            case 400:
              console.error("Bad request. Please check the input data.");
              break;
            case 403:
              console.error("Authorization error. Please check the access token.");
              break;
            case 500:
              console.error("Server error. Please try again later.");
              break;
            default:
              console.error(`Unexpected error: ${error.response.status}`);
          }
        } else {
          console.error("Network error or server is not responding.");
        }
      } finally {
        setIsSending(false);
      }
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <InputContainer>
      <TextInput
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={isSending ? "" : "메시지를 입력하세요..."}
        disabled={isSending}
      />
      <SendButton onClick={handleSend} disabled={isSending}>
        {isSending ? <LoadingSpinner /> : <img src={inputButton} alt="Send" />}
      </SendButton>
    </InputContainer>
  );
};

export default InputArea;