import React, { useState } from "react";
import axios from "axios";
import './App.css';

function App() {
  const [mediaFile, setMediaFile] = useState(null);
  const [title, setTitle] = useState("");
  const [transcription, setTranscription] = useState("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");

  const handleFileChange = (e) => {
    setMediaFile(e.target.files[0]);
  };

  const handleTitleChange = (e) => {
    setTitle(e.target.value);
  };

  const handleQuestionChange = (e) => {
    setQuestion(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!mediaFile || !title) {
      alert("Please provide both a title and a media file.");
      return;
    }

    const formData = new FormData();
    formData.append("media", mediaFile);
    formData.append("title", title);

    try {
      const response = await axios.post("http://127.0.0.1:9090/audioToText", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      setTranscription(response.data.transcription);
    } catch (error) {
      console.error("Error transcribing audio:", error);
    }
  };

  const handleQuestionSubmit = (e) => {
    e.preventDefault();

    if (!question || !transcription) {
      alert("Please provide both a question and a transcription.");
      return;
    }

    const eventSource = new EventSource(`http://127.0.0.1:9090/answer_question?question=${encodeURIComponent(question)}&transcription=${encodeURIComponent(transcription)}`);

    eventSource.onmessage = (event) => {
      if (event.data === "[DONE]") {
        eventSource.close();
      } else {
        setAnswer((prev) => prev + event.data);
      }
    };

    eventSource.onerror = (error) => {
      console.error("EventSource failed:", error);
      eventSource.close();
      setAnswer((prev) => prev + "\nError: Connection failed.");
    };
  };

  return (
    <div className="App">
      <h1>Upload Media File</h1>
      <form onSubmit={handleSubmit}>
        <div>
          <label>Title: </label>
          <input type="text" value={title} onChange={handleTitleChange} />
        </div>
        <div>
          <label>Media File: </label>
          <input type="file" onChange={handleFileChange} />
        </div>
        <button type="submit">Submit</button>
      </form>
      {transcription && (
        <div>
          <h2>Transcription</h2>
          <p>{transcription}</p>
          <h2>Ask a Question</h2>
          <form onSubmit={handleQuestionSubmit}>
            <textarea rows="4" cols="50" value={question} onChange={handleQuestionChange} />
            <br />
            <button type="submit">Submit</button>
          </form>
          <div>
            <h2>Answer</h2>
            <pre>{answer}</pre>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
