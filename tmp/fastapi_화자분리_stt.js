import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [file, setFile] = useState(null);
  const [numSpeakers, setNumSpeakers] = useState('');
  const [transcriptions, setTranscriptions] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleSpeakersChange = (event) => {
    setNumSpeakers(event.target.value);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!file || !numSpeakers) {
      alert('Please fill in all fields');
      return;
    }

    const formData = new FormData();
    formData.append('media', file);
    formData.append('num_speakers', numSpeakers);

    setLoading(true);
    try {
      const response = await axios.post('http://localhost:8000/extract_text', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      setTranscriptions(response.data.transcriptions);
    } catch (error) {
      console.error('Error:', error);
      alert('Error processing your request');
    }
    setLoading(false);
  };

  return (
    <div className="App">
      <h1>Audio Transcription App</h1>
      <form onSubmit={handleSubmit}>
        <label>
          Upload audio file:
          <input type="file" onChange={handleFileChange} />
        </label>
        <label>
          Number of speakers:
          <input type="number" value={numSpeakers} onChange={handleSpeakersChange} />
        </label>
        <button type="submit" disabled={loading}>
          {loading ? 'Processing...' : 'Submit'}
        </button>
      </form>
      <div>

 {/* Transcriptions 이 배열로 들어올 경우 */}

        <h2>Transcriptions</h2>
        {transcriptions.length > 0 ? (
          <ul>
            {transcriptions.map((transcription, index) => (
              <li key={index}>{transcription}</li>
            ))}
          </ul>
        ) : (
          <p>No transcriptions available.</p>
        )}

 {/* Transcriptions 이 문자열로 들어올 경우 */}

        {/* <h2>Transcriptions</h2>
          {transcriptions ? (
            <p>{transcriptions}</p>
          ) : (
            <p>No transcription available.</p>
          )} */}
      </div>
    </div>
  );
}

export default App;
