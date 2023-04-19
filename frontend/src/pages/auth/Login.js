import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import axios from "axios";
import GoogleButton from "react-google-button";
import "./vendor/login/bootstrap/css/bootstrap.min.css";
import "./fonts/font-awesome-4.7.0/css/font-awesome.min.css";
import "./fonts/Linearicons-Free-v1.0.0/icon-font.min.css";
import "./vendor/login/animate/animate.css";
import "./vendor/login/css-hamburgers/hamburgers.min.css";
import "./vendor/login/animsition/css/animsition.min.css";
import "./vendor/login/select2/select2.min.css";
import "./vendor/login/daterangepicker/daterangepicker.css";
import "./css/login/util.css";
import "./css/login/main.css";
import  socket  from '../../socket';
import { signInWithEmailAndPassword } from "firebase/auth";
import {auth } from '../../firebase';

export default function Login(props) {
  const navigate = useNavigate()
  const [state, setState] = useState({
    email: "",
    password: "",
  });
  const handleChange = (e) => {
    setState((state) => ({
      ...state,
      ...{ [e.target.name]: e.target.value },
    }));
    // console.log(state);
  };
  const [role, setRole] = useState("User");
  const [token, setToken] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [user, setUser] = useState({});
  const SubmitMongo = async () => {
    await axios.post('http://localhost:8000/login', {
      email: state.email,
    })
      .then((response) => {
        localStorage.setItem('user', JSON.stringify(response.data));
        setUser(response.data)
        console.log(response.data);
        if(response.data.role === 'collector'){
          socket.emit('join_room', 'room2')
          if(response.data.status === 'unverified')
          {
            alert('Please wait for the admin to approve your request for signing in as a collector and login after sometime')
            navigate('/');
          }
          else
          {
            navigate('/collector')
          }
        }
        else if(response.data.role === 'User') {
          socket.emit('join_room', 'room1')
          navigate('/dashboard')
        }
        else
        {
          navigate('/admin')
        }
      })
      .catch((error) => {
        setErrorMessage(error);
        console.log(error);
      });
  }

    

  const handleRoleChange = (event) => {
    setRole(event.target.value);
  };
  const handleSubmit = async(e) => {
    e.preventDefault();

    // console.log(state);
    await signInWithEmailAndPassword(auth, state.email, state.password)
      .then((userCredential) => {
        setToken(userCredential.user.accessToken)
        // setErrorMessage('User signed successfully')
        localStorage.setItem('token', JSON.stringify(userCredential.user.accessToken));
        console.log(userCredential.user.accessToken);
        SubmitMongo()
      })
      .catch((error) => {
        console.log(error);
        setErrorMessage(error.message);
      });
  };
  return (
    <div>
      <div className="limiter">
        <div
          className="container-login100"
          style={{
            backgroundColor: "white",
          }}
        >
          <div className="wrap-login100 p-t-30 p-b-50">
            {/* <span className="login100-form-title p-b-41">Account Login</span> */}
            <form
              className="login100-form validate-form p-b-33 p-t-5"
              action="/"
              onSubmit={handleSubmit}
            >
              {/* <div className="container-login100-form-btn m-t-16">
                <GoogleButton
                  onClick={() => {
                    console.log("Google button clicked");
                  }}
                />
              </div> */}
              {/* <div className="container-login100-form-btn m-t-16">
                <h3>OR</h3>
              </div> */}
              <div className="container-login100-form-btn m-t-16 mb-5">
                <h2>ACCOUNT LOGIN</h2>
              </div>
              {/* <div className="ms-5" style={{ width: "50%", marginLeft: "12%" }}>
                <label>Select Role</label>
                <select
                  className="form-select form-select-sm"
                  aria-label=".form-select-sm example"
                  onChange = {handleRoleChange}
                >
                  <option value="user">User</option>
                  <option value="collector">Collector</option>
                </select>
              </div> */}
              <div
                className="wrap-input100 validate-input"
                data-validate="Enter username"
              >
                <input
                  className="input100"
                  type="text"
                  name="email"
                  placeholder="Email"
                  onChange={handleChange}
                />
                <span
                  className="focus-input100"
                  data-placeholder="&#xe82a;"
                ></span>
              </div>

              <div
                className="wrap-input100 validate-input"
                data-validate="Enter password"
              >
                <input
                  className="input100"
                  type="password"
                  name="password"
                  placeholder="Password"
                  onChange={handleChange}
                />
                <span
                  className="focus-input100"
                  data-placeholder="&#xe80f;"
                ></span>
              </div>
              {errorMessage && (
                  <div> {errorMessage}</div> 
                )}
              <div className="container-login100-form-btn m-t-32">
                <button className="login100-form-btn" type="submit">
                  Login
                </button>
              </div>
              <div className="container-login100-form-btn m-t-10">
                <Link to="/register">
                  <p>Register now</p>
                </Link>
              </div>
            </form>
          </div>
        </div>
      </div>

      <div id="dropDownSelect1"></div>

      <script src="./vendor/login/jquery/jquery-3.2.1.min.js" />
      <script src="./vendor/login/animsition/js/animsition.min.js" />
      <script src="./vendor/login/bootstrap/js/popper.js" />
      <script src="./vendor/login/bootstrap/js/bootstrap.min.js" />
      <script src="./vendor/login/select2/select2.min.js" />
      <script src="./vendor/login/daterangepicker/moment.min.js" />
      <script src="./vendor/login/daterangepicker/daterangepicker.js" />
      <script src="./vendor/login/countdowntime/countdowntime.js" />
      <script src="./js/main.js" />
    </div>
  );
}
