import styled from 'styled-components';
import { GoogleLogin } from '@react-oauth/google';

export const LoginPageContainer = styled.div`
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100vh;
    font-family: 'Roboto', sans-serif;
`;

export const LoginTitle = styled.h1`
    margin-bottom: 20px;
`;

export const GoogleLoginButton = styled(GoogleLogin)`
    font-size: 16px;
    font-weight: bold;
`;