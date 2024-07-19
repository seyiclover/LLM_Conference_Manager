import React, { useState } from 'react';
import Calendar from 'react-calendar';
import 'react-calendar/dist/Calendar.css';
import styled from 'styled-components';

const CalendarContainer = styled.div`
  .react-calendar {
    width: 100%;
    border: none;
    font-family: Arial, Helvetica, sans-serif;
    line-height: 1.125em;
    margin-top: 20px;
    background-color: #f0f0f0;
  }

  .react-calendar__tile {
    max-width: 100%;
    padding: 10px 6.6667px;
    background: none;
    text-align: center;
    line-height: 16px;
    border-radius: 4px;
    border: 0;
    outline: none;

    &:hover {
      background-color: #f0f0f0;
    }
  }

  .react-calendar__tile--active {
    background: #0281F2;
    color: white;
  }

  .react-calendar__navigation button {
    color: #0281F2;
    min-width: 44px;
    background: none;
    font-size: 16px;
    margin-top: 8px;

    &:disabled {
      background-color: #f0f0f0;
    }

    &:hover:not(:disabled) {
      background-color: #e6e6e6;
    }
  }
`;

const MyCalendar = () => {
  const [date, setDate] = useState(new Date());

  const onChange = (date) => {
    setDate(date);
  };

  return (
    <CalendarContainer>
      <Calendar onChange={onChange} value={date} />
    </CalendarContainer>
  );
};

export default MyCalendar;
