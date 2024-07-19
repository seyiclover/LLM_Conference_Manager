import styled from 'styled-components';
import { Link } from 'react-router-dom';

export const MainPageContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 100%;
  margin: 0 auto;
  padding: 0;
  font-family: 'Roboto', sans-serif;
`;

export const Navbar = styled.div`
  background-color: #f8f9fa;
  display: flex;
  align-items: center;
  position: fixed;
  top: 0;
  width: 100%;
  z-index: 1000;
  height: 5rem;
  justify-content: space-between;
  box-shadow: 0 0.25rem 0.25rem rgba(0, 0, 0, 0.4);
  margin-bottom: 0;

  img {
    margin-left: 4.6875rem;
    width: 5.625rem;
    height: 5.2rem;
    padding-top: 0.625rem;
  }

  a {
    text-decoration: none;
    color: #000;
    margin-right: 1.25rem;

    &:hover {
      text-decoration: underline;
    }
  }
`;

export const NavText = styled.span`
  font-weight: 900;
  font-size: 2.5rem;
  color: #05305b;
  position: absolute;
  left: 10rem;
  top: 1rem;
`;

export const NavRight = styled.div`
  margin-left: auto;
  display: flex;
  align-items: center;
`;

export const LoginLink = styled(Link)`
  color: #3788b4 !important;
  font-family: 'Roboto';
  font-weight: 700;
  font-size: 1.25rem;
  margin-right: 2.188rem !important;
`;

export const SignUpLink = styled(Link)`
  color: #ffffff !important;
  background-color: #3788b4;
  padding: 0.3125rem 4.375rem;
  font-family: 'Roboto';
  font-weight: 700;
  font-size: 1.25rem;
  margin-right: 4.375rem !important;
`;

export const LogoutButton = styled.button`
  color: #ffffff !important;
  background-color: #3788b4;
  padding: 0.3125rem 4.375rem;
  font-family: 'Roboto';
  font-weight: 700;
  font-size: 1.25rem;
  margin-right: 3.125rem;
  border: none;
  cursor: pointer;
`;

export const MainSection = styled.div`
  display: flex;
  flex-direction: row;
  align-items: flex-start;
  justify-content: center;
  width: 100%;
  margin: 0 auto;
  padding-top: 3.8rem;
  box-sizing: border-box;

`;

export const LeftSection = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  position: relative;
  padding-top: 2rem;

  text-align: center;
  margin-top: 0rem;

  p {
    margin-top: 6.813rem;
    font-weight: 900;
    font-size: 2.1875rem;
    color: #05305b;
    z-index: 2;
    margin-bottom: 0;
    margin-right: 10rem;
  }

  .image-container {
    display: flex;
    flex-direction: row;
    justify-content: center;
    align-items: center;
    margin-right: 10rem;

    img {
      margin-right: 1rem;
      height: auto;
      max-width: 100%;
      z-index: 2;
    }
  }

  img.robot {
    max-width: 26.1875rem;
    width: 100%;
    z-index: 3;
    margin-right: 13rem;
  }

  img.circle {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: auto;
    z-index: 1;
  }
`;

export const RightContainer = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-top: 7rem;
`;

export const TitleSection = styled.div`
  width: 100%;
  text-align: center;
  h1 {
    font-weight: 900;
    color: #05305b;
    font-family: 'Roboto Condensed';
    font-size: 8rem;
    font-style: normal;
    line-height: 9.125rem;
    height: 7.3rem;
    letter-spacing: -0.1rem;
  }
  p {
    font-weight: 900;
    font-size: 2rem;
    color: #000000;
    margin-top: -5.2rem;
  }
`;

export const CTAButton = styled(Link)`
  background-color: #3788b4;
  color: #ffffff;
  padding: 1.25rem;
  border: none;
  border-radius: 1.25rem;
  font-size: 1.5625rem;
  font-weight: 700;
  cursor: pointer;
  margin-top: 2rem;
  width: 11.625rem;
  text-decoration: none;
  display: inline-block;
  text-align: center;

`;

export const Partners = styled.div`
  width: 100%;
  display: flex;
  justify-content: center;
  margin-top: 2rem;
  margin-right: 2rem;

  img {
    width: 12.5rem;
    object-fit: contain;
    height: 13rem;
  }
  img:nth-child(n+2) {
    margin-left: 1.5rem;
  }


  .zoomClova {
    transform: scale(1.5);
  }
  .zoomNotion {
    transform: scale(1.2);
  }
`;