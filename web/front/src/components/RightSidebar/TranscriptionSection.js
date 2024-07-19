import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import axios from 'axios';
import arrow from '../../images/arrowback.png';

const TranscriptionSectionContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
`;

const Header = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  padding: 10px 20px;
  margin-top: 10px;
`;

const BackButton = styled.button`
  border: none;
  background: none;
  cursor: pointer;
  margin: 0px;
  padding: 0px;
`;

const SectionTitle = styled.h2`
  flex-grow: 1;
  text-align: left;
  margin: 0px 0px 6px 0px;
  font-family: 'Roboto Condensed', sans-serif;
  font-style: normal;
  font-weight: 700;
  font-size: 20px;
  line-height: 23px;
  color: #000000;
`;

const ContentSection = styled.div`
  width: 100%;
  max-width: 600px;
  padding: 0px 20px;
  border-radius: 10px;
`;

const InfoSection = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  margin-bottom: 15px;
  border-bottom: 1px solid #D9D9D9;
  padding-bottom: 3px;
`;

const InfoItemContainer = styled.div`
  display: flex;
  align-items: center;
  flex-grow: 1;
`;

const InfoItem = styled.div`
  font-size: 16px;
  color: #555555;
  margin-right: 10px;
  font-style: normal;
  font-weight: 400;
  font-size: 13px;
  font-family: 'Abel', sans-serif;
`;

const TranscriptList = styled.ul`
  list-style: none;
  padding: 0;
  margin: 0;
`;

const TranscriptItem = styled.li`
  margin-bottom: 10px;
  font-family: 'Poppins', sans-serif;
  font-style: normal;
  font-weight: 400;
  font-size: 15px;
  line-height: 21px;
  letter-spacing: -0.01em;
  color: #000000;
`;

const Speaker = styled.span`
  font-weight: bold;
  color: #3788B4;
`;

const ToggleContainer = styled.div`
  display: flex;
  align-items: center;
  padding-bottom: 2px;
`;

const ToggleLabel = styled.label`
  font-size: 13px;
  margin-right: 10px;
  color: #555555;
`;

const ToggleSwitch = styled.input`
  position: relative;
  width: 38px;
  height: 20px;
  background-color: #ccc;
  outline: none;
  border-radius: 20px;
  transition: 0.4s;
  cursor: pointer;
  -webkit-appearance: none;

  &:checked {
    background-color: #0281F2;
  }

  &:before {
    position: absolute;
    content: '';
    width: 18px;
    height: 20px;
    border-radius: 50%;
    background-color: #fff;
    transition: 0.4s;
    transform: ${props => props.checked ? 'translateX(20px)' : 'translateX(0)'};
  }
`;

// const SummaryText = styled.p`
//   font-family: 'Poppins', sans-serif;
//   font-style: normal;
//   font-weight: 400;
//   font-size: 15px;
//   line-height: 21px;
//   letter-spacing: -0.01em;
//   color: #000000;
// `;

const SummaryItem = styled.div`
  margin-bottom: 10px;
  font-family: 'Poppins', sans-serif;
  font-style: normal;
  font-weight: 400;
  font-size: 15px;
  line-height: 21px;
  letter-spacing: -0.01em;
  color: #000000;
`;

const TranscriptionSection = ({ meetingId, onBackClick }) => {
  const [transcription, setTranscription] = useState(null);
  const [summary, setSummary] = useState('');
  const [isSummaryOn, setIsSummaryOn] = useState(false);

  useEffect(() => {
    const fetchTranscription = async () => {
      try {
        const response = await axios.get(`http://localhost:8000/upload/meeting/${meetingId}`, {
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${localStorage.getItem('access_token')}`
          }
        });
        console.log(response.data);  // 응답 데이터 확인
        setTranscription(response.data);
      } catch (error) {
        console.error('Error fetching transcription:', error);
      }
    };

    fetchTranscription();
  }, [meetingId]);

  const handleToggleChange = async () => {
    setIsSummaryOn(!isSummaryOn);
    if (!isSummaryOn) {
      try {
        const response = await axios.post(`http://localhost:8000/clova/summarize`, {
            text: transcription.content,
            transcript_id: meetingId,
        }, {
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${localStorage.getItem('access_token')}`
          }
        });
        setSummary(response.data.summary);
      } catch (error) {
        console.error('Error fetching summary:', error);
      }
    }
  };

  if (!transcription) {
    return <p>Loading...</p>;
  }

  const transcriptItems = transcription.content.split('\n').map((line, index) => {
    const [speaker, ...speech] = line.split(': ');
    return (
      <TranscriptItem key={index}>
        <Speaker>{speaker}:</Speaker> {speech.join(': ')}
      </TranscriptItem>
    );
  });

  const summaryItems = summary.split('- ').filter(item => item.trim() !== '').map((item, index) => (
    <SummaryItem key={index}>{item}</SummaryItem>
  ));

  return (
    <TranscriptionSectionContainer>
      <Header>
        <BackButton onClick={onBackClick}><img src={arrow} alt="back" /></BackButton>
      </Header>
      <ContentSection>
        <SectionTitle>{transcription.title}</SectionTitle>
        <InfoSection>
          <InfoItemContainer>
            <InfoItem>회의 날짜: {transcription.date}</InfoItem>
            <InfoItem>참석자: {transcription.num_speakers}</InfoItem>
          </InfoItemContainer>
          <ToggleContainer>
            <ToggleLabel>요약</ToggleLabel>
            <ToggleSwitch
              type="checkbox"
              checked={isSummaryOn}
              onChange={handleToggleChange}
            />
          </ToggleContainer>
        </InfoSection>
        {isSummaryOn ? (
          summaryItems
        ) : (
          <TranscriptList>
            {transcriptItems}
          </TranscriptList>
        )}
      </ContentSection>
    </TranscriptionSectionContainer>
  );
};

export default TranscriptionSection;
