import React, { useState, useEffect } from 'react';
import axios from 'axios';
import styled from 'styled-components';
import { useLocation } from 'react-router-dom'; // useNavigate로 변경
import TranscriptionSection from './TranscriptionSection';
import UploadModal from './UploadModal';
import Calendar from './Calendar'; // Calendar 컴포넌트 import

const SidebarContainer = styled.div`
  width: 400px;
  background-color: #f0f0f0;
  padding: 20px;
  position: absolute;
  right: 10px;
  top: 15px;
  bottom: 10px;
  height: calc(100vh - 10px);
  display: flex;
  flex-direction: column;
  border-radius: 10px 30px 30px 10px;
  box-sizing: border-box;

  ::-webkit-scrollbar {
    width: 0px;
  }
`;

const Tabs = styled.div`
  display: flex;
  justify-content: space-around;
  margin-top: 15px;
`;

const Tab = styled.button`
  background: none;
  border: none;
  position: relative;
  padding: 10px;
  font-size: 14px;
  cursor: pointer;
  color: #333;
  font-weight: normal;
  flex: 1;

  &::after {
    content: '';
    position: absolute;
    bottom: -5px;
    left: 0;
    width: 100%;
    height: 2px;
    background-color: ${(props) => (props.$active ? '#0281F2' : 'transparent')};
  }

  &:hover::after {
    background-color: ${(props) => (props.$active ? '#0281F2' : '#ddd')};
  }

  ${(props) => props.$active && `
    color: #4285F4;
    font-weight: bold;
  `}
`;

const TabContent = styled.div`
  flex: 4;
  overflow-y: auto;
`;

const UploadSection = styled.div`
  align-items: center;
  justify-content: center;
  display: flex;
  flex-direction: column;
`;

const CalendarSection = styled.div`
  align-items: center;
  justify-content: center;
`;

const MeetingListSection = styled.div`
  flex: 3;
  overflow-y: auto;
`;

const SectionTitle = styled.h2`
  font-size: 20px;
  margin-bottom: 15px;
  color: #05305B;
`;

const MeetingList = styled.ul`
  list-style: none;
  padding: 0;
  margin: 0;
`;

const MeetingItem = styled.li`
  background-color: #fff;
  padding: 15px;
  border-radius: 15px;
  margin-bottom: 10px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
  cursor: pointer;
`;

const MeetingTitle = styled.span`
  font-weight: regular;
  color: #05305B;
  font-size: 20px;
`;

const MeetingDate = styled.span`
  font-size: 14px;
  color: #666;
  margin-left: 10px;
`;

const StyledParagraph = styled.p`
  font: Roboto Condensed;
  &:first-child {
    margin-top: 30px;
    font-size: 20px;
    color: #05305B;
    font-weight: bold;
  }
  &:last-child {
    font-size: 20px;
    font-weight: regular;
  }
`;

const SectionDivider = styled.div`
  width: 100%;
  height: 2px;
  background-color: #e0e0e0;
  margin: 10px 0;
`;

const UBTNSidebar = styled.button`
  background: none;
  border: 2px solid #0073e6;
  border-radius: 10px;
  padding: 20px;
  cursor: pointer;
  font-size: 16px;
  color: #0073e6;
  margin-top: 10px;
  align-self: flex-end;
`;

const RightSidebar = ({ userId }) => {
  const [activeTab, setActiveTab] = useState('upload');
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [meetings, setMeetings] = useState([]);
  const [selectedMeetingId, setSelectedMeetingId] = useState(null);
  const location = useLocation();
  //const navigate = useNavigate();  // useNavigate 훅 사용

  const fetchMeetings = async () => {
    try {
      const response = await axios.get('http://localhost:8000/upload/listMeetings', {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('access_token')}` // 인증 토큰 추가
        }
      });
      setMeetings(response.data.meetings);
    } catch (error) {
      console.error('Error fetching meetings:', error);
    }
  };

  useEffect(() => {
    fetchMeetings();
  }, [location]);

  const handleUploadClick = () => {
    setIsModalOpen(true);
  };

  const closeModal = () => {
    setIsModalOpen(false);
  };

  const handleUpload = (response) => {
    const meetingId = response?.meeting_id;
    if (meetingId) {
      console.log(meetingId);
      setSelectedMeetingId(meetingId); // Prop으로 전달
      setIsModalOpen(false); // 모달 닫기
      //navigate(`/upload/meeting/${meetingId}`);  // 상세 페이지로 리다이렉션
    } else {
      console.error("meeting_id가 응답에 포함되어 있지 않습니다.", response);
    }
  };



  const handleBackClick = () => {
    setSelectedMeetingId(null);
    fetchMeetings();
  };

  return (
    <SidebarContainer>
      <Tabs>
        <Tab $active={activeTab === 'upload'} onClick={() => setActiveTab('upload')}>음성업로드</Tab>
        <Tab $active={activeTab === 'calendar'} onClick={() => setActiveTab('calendar')}>일정표</Tab>
      </Tabs>
      <TabContent>
        {activeTab === 'upload' && selectedMeetingId ? (
          <TranscriptionSection meetingId={selectedMeetingId} onBackClick={handleBackClick} />
        ) : (
          <>
            {activeTab === 'upload' ? (
              <UploadSection>
                <StyledParagraph>파일 업로드로 회의록을 텍스트 변환하고 요약해서 사용하세요!</StyledParagraph>
                <StyledParagraph>파일 업로드 시 먼저 회의록에 날짜와 제목 그리고 발화자 수를 선택해주세요</StyledParagraph>
                <UBTNSidebar onClick={handleUploadClick}>파일 첨부</UBTNSidebar>
              </UploadSection>
            ) : (
              <CalendarSection>
                <Calendar />
              </CalendarSection>
            )}
            {activeTab === 'upload' && (
              <MeetingListSection>
                <SectionDivider />
                <SectionTitle>회의 목록</SectionTitle>
                {Array.isArray(meetings) && meetings.length === 0 ? (
                  <StyledParagraph>음성업로드를 하여 회의록을 추가하세요!</StyledParagraph>
                ) : (
                  <MeetingList>
                    {Array.isArray(meetings) && [...meetings].reverse().map((meeting, index) => (
                      <MeetingItem key={meeting.id} onClick={() => {
                        setSelectedMeetingId(meeting.id);
                      }}>
                        <div>
                          <MeetingTitle>{meeting.filename.replace(/\.[^/.]+$/, "")}</MeetingTitle>
                          <MeetingDate>{meeting.meeting_date}</MeetingDate>
                        </div>
                      </MeetingItem>
                    ))}
                  </MeetingList>
                )}
              </MeetingListSection>
            )}
          </>
        )}
      </TabContent>
      <UploadModal isOpen={isModalOpen} onClose={closeModal} onUpload={handleUpload} /> {/* 모달 열기 */}
    </SidebarContainer>
  );
};

export default RightSidebar;