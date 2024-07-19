import React from 'react';
import styled from 'styled-components';
import PropTypes from 'prop-types';

const UploadButton = styled.button`
  background: none;
  border: 2px dashed #0073e6;
  border-radius: 10px;
  padding: 20px;
  cursor: pointer;
  font-size: 16px;
  color: #0073e6;
  margin-top: 20px;
  position: relative;
  right: 0px;
`;

const FileInput = styled.input`
  display: none;
`;

const VoiceUpload = ({ onFileChange }) => {
  const fileInputRef = React.createRef();

  const handleButtonClick = () => {
    fileInputRef.current.click();
  };

  const handleChange = (event) => {
    const file = event.target.files[0];
    if (onFileChange) {
      onFileChange(file);
    }
  };

  return (
    <>
    
      <UploadButton onClick={handleButtonClick}>파일 첨부</UploadButton>
      {/* 오디오 파일만 받기 */}
      <FileInput
        type="file"
        accept="audio/*"
        ref={fileInputRef}
        onChange={handleChange}
      />
    </>
  );
};

VoiceUpload.propTypes = {
  onFileChange: PropTypes.func.isRequired,
};

export default VoiceUpload;
