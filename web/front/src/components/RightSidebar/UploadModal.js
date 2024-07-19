import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import axios from 'axios';
import VoiceUpload from './VoiceUpload'; // VoiceUpload 컴포넌트 임포트
import Spinner from '../../utils/spinner'; // Spinner 컴포넌트 임포트

const ModalOverlay = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.5); 
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
`;

const ModalContent = styled.div`
  width: 400px;
  background: white;
  padding: 30px;
  border-radius: 8px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  z-index: 1001;
  position: relative;
`;

const ModalHeader = styled.h2`
  text-align: center;
  margin-bottom: 20px;
  font-weight: bold;
`;

const ModalForm = styled.form`
  display: flex;
  flex-direction: column;
`;

const ModalLabel = styled.label`
  margin-top: 10px;
  margin-bottom: 5px;
  font-weight: bold;
`;

const ModalInput = styled.input`
  padding: 10px;
  border: 1px solid #ccc;
  border-radius: 5px;
  margin-bottom: 10px;
  font-size: 14px;
`;

const ModalSelect = styled.select`
  padding: 10px;
  border: 1px solid #ccc;
  border-radius: 5px;
  margin-bottom: 10px;
  font-size: 14px;
`;

const ModalFooter = styled.div`
  display: flex;
  justify-content: flex-end;
  margin-top: 20px;
`;

const CloseButton = styled.button`
  background-color: #ccc;
  color: black;
  border: none;
  padding: 10px 20px;
  cursor: pointer;
  border-radius: 5px;
  margin-right: 10px;
`;

const SaveButton = styled.button`
  background-color: #007bff;
  color: white;
  border: none;
  padding: 10px 20px;
  cursor: pointer;
  border-radius: 5px;
  position: relative;
`;

const LoadingOverlay = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.7); 
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1001;
  flex-direction: column;
`;

const LoadingMessage = styled.div`
  color: white;
  margin-top: 20px;
  font-size: 18px;
`;

const UploadModal = ({ isOpen, onClose, onUpload }) => {
  const [meetingName, setMeetingName] = useState('');
  const [meetingDate, setMeetingDate] = useState('');
  const [numSpeakers, setNumSpeakers] = useState(1);
  const [file, setFile] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (isOpen) {
      // 모달이 열릴 때 초기화
      setMeetingName('');
      setMeetingDate('');
      setNumSpeakers(1);
      setFile(null);
      setError('');
      setLoading(false);
    }
  }, [isOpen]);

  const handleFileChange = (selectedFile) => {
    setFile(selectedFile);
    setError(''); // 파일이 선택되면 에러 메시지 초기화
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    if (!file) {
      setError('파일을 선택해주세요.');
      return;
    }
    setLoading(true); // 로딩 상태 활성화
    const formData = new FormData();
    formData.append('file', file);
    formData.append('meeting_name', meetingName);
    formData.append('meeting_date', meetingDate);
    formData.append('speaker_count', numSpeakers);

    try {
      // 파일 업로드 요청
      const uploadResponse = await axios.post('http://localhost:8000/upload/files', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          'Authorization': `Bearer ${localStorage.getItem('access_token')}` // 인증 토큰 추가
        },
      });

      if (uploadResponse.status === 200 && uploadResponse.data.id) {
        const fileId = uploadResponse.data.id.toString();
        console.log('File uploaded successfully with ID:', fileId);

        // 파일 업로드가 성공하면 변환 요청을 원격 서버에 보냄
        const sttResponse = await axios.post('http://localhost:8000/upload/audioToText', {
          file_id: fileId,
          num_speakers: numSpeakers,
          title: meetingName,
          meeting_date: meetingDate,
        }, {
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${localStorage.getItem('access_token')}`
          }
        });
        console.log('STT Response:', sttResponse.data);

        const meetingId = sttResponse.data.meeting_id;
        if (meetingId) {
          console.log('Meeting ID:', meetingId);
          onUpload(sttResponse.data); // 변환 후 필요한 데이터를 상위 컴포넌트로 전달
          setLoading(false); // 로딩 상태 비활성화
          onClose(); // 성공 시 모달 닫기
        } else {
          console.error('meeting_id가 응답에 포함되어 있지 않습니다.', sttResponse.data);
          setLoading(false);
        }
      } else {
        setError('File upload failed. Please try again.');
        console.error('File upload failed:', uploadResponse.data);
        setLoading(false);
      }
    } catch (error) {
      setError('Error uploading file or transcribing. Please try again.');
      console.error('Error uploading file or transcribing:', error);

      if (error.response) {
        console.error('Error response data:', error.response.data);
      } else {
        console.error('Error details:', error.message);
      }

      setLoading(false); // 에러 발생 시 로딩 상태 비활성화
    }
  };

  if (!isOpen) return null;

  return (
    <>
      <ModalOverlay>
        <ModalContent>
          <ModalForm onSubmit={handleSubmit}>
            <ModalHeader>회의 정보 입력</ModalHeader>
            {error && <p style={{ color: 'red' }}>{error}</p>}
            <ModalLabel>회의 제목:</ModalLabel>
            <ModalInput
              type="text"
              value={meetingName}
              onChange={(e) => setMeetingName(e.target.value)}
              required
            />
            <ModalLabel>날짜:</ModalLabel>
            <ModalInput
              type="date"
              value={meetingDate}
              onChange={(e) => setMeetingDate(e.target.value)}
              required
            />
            <ModalLabel>참석자 수:</ModalLabel>
            <ModalSelect
              value={numSpeakers}
              onChange={(e) => setNumSpeakers(parseInt(e.target.value))}
              required
            >
              <option value="1">1명</option>
              <option value="2">2명</option>
              <option value="3">3명</option>
              <option value="4">4명</option>
              <option value="5">5명</option>
            </ModalSelect>
            <ModalLabel>파일 업로드:</ModalLabel>
            <VoiceUpload onFileChange={handleFileChange} />
            <ModalFooter>
              <CloseButton type="button" onClick={onClose}>닫기</CloseButton>
              <SaveButton type="submit" disabled={!file || loading}>
                업로드
              </SaveButton>
            </ModalFooter>
          </ModalForm>
        </ModalContent>
      </ModalOverlay>
      {loading && (
        <LoadingOverlay>
          <Spinner />
          <LoadingMessage>음성을 텍스트로 전사 중입니다. 시간이 조금 걸릴 수 있습니다...</LoadingMessage>
        </LoadingOverlay>
      )}
    </>
  );
};

export default UploadModal;