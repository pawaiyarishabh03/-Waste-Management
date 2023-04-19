import React, { useState } from "react";
import axios from "axios";
import "./vendor/register/mdi-font/css/material-design-iconic-font.min.css";
import "./vendor/register/font-awesome-4.7/css/font-awesome.min.css";
import "./vendor/register/select2/select2.min.css";
import "./vendor/register/datepicker/daterangepicker.css";
import "./css/register/main.css";
import { createUserWithEmailAndPassword } from "firebase/auth";
import { auth } from "../../firebase";
import {useNavigate} from 'react-router-dom'
import socket from '../../socket'
export default function Register() {
  const navigate = useNavigate();
  const [state, setState] = useState({
    name: "",
    email: "",
    phone: "",
    password: "",
    confirmpassword: "",
  });
  const [role, setRole] = useState("User");
  const [token, setToken] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [user, setUser] = useState(null);
  const handleRoleChange = (event) => {
    console.log(event.target.value);
    setRole(event.target.value);
  };
  const handleChange = (e) => {
    // console.log(state);
    setState((state) => ({
      ...state,
      ...{ [e.target.name]: e.target.value },
    }));
    // console.log(state);
  };

  const submitMongo = async () => {
    await axios
      .post("http://localhost:8000/register", {
        name: state.name,
        email: state.email,
        number: state.phone,
        role: role,
      })
      .then((response) => {
        console.log(response);
        localStorage.setItem("user", JSON.stringify(response.data));
        setUser(response.data);
        if(response.data.role === 'collector'){
          socket.emit('join_room', 'room2')
          alert('Please wait for the admin to approve your request for signing in as a collector and login after sometime')
          navigate('/');
        }
        else{
          socket.emit('join_room', 'room1')
        }
        navigate('/dashboard')
      })
      .catch((error) => {
        if (error.response) {
            // Request made and server responded
            setErrorMessage(error.response.data);
            console.log(error.response.status);
            console.log(error.response.headers);
        } else if (error.request) {
            // The request was made but no response was received
            setErrorMessage(error.request);
        } else {
            // Something happened in setting up the request that triggered an Error
            setErrorMessage(error.message);
        }
      });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    console.log("entered handleSubmit");
   await createUserWithEmailAndPassword(auth, state.email, state.password)
      .then((userCredential) => {
        setToken(userCredential.user.accessToken);
        // setErrorMessage("User signed successfully");
        localStorage.setItem(
          "token",
          JSON.stringify(userCredential.user.accessToken)
        );
        localStorage.setItem("role", JSON.stringify(role));
        // console.log(info)

        submitMongo();
      })
      .catch((error) => {
        // const errorCode = error.code;
        // console.log(errorCode);
        // console.log(error.message)
        const errorMessage = error.message;
        setErrorMessage(errorMessage);

        console.log(error);
      });
  };
  return (
    <div>
      <div class="page-wrapper bg-gra-02 p-t-130 p-b-100 font-poppins">
        <div class="wrapper wrapper--w680">
          <div class="card card-4">
            <div class="card-body">
              <h2 class="title">Registration Form</h2>
              <form onSubmit={handleSubmit}>
                <div class="row row-space">
                  <div class="col-12 mb-3">
                    <label>Select Role</label>
                    <select
                      class="form-select form-select-sm"
                      aria-label=".form-select-sm example"
                      onChange={handleRoleChange}
                    >
                      <option value="user">User</option>
                      <option value="collector">Collector</option>
                    </select>
                  </div>
                </div>
                <div class="row row-space">
                  <div class="col-12">
                    <div class="input-group">
                      <label class="label" for="name">
                        Name
                      </label>
                      <input style ={{ lineHieght : "25px" }}  
                        class="input--style-4"
                        type="text"
                        name="name"
                        onChange={handleChange}
                      />
                    </div>
                  </div>
                </div>
                <div class="row row-space">
                  <div class="col-12">
                    <div class="input-group">
                      <label class="label" for="email">
                        Email
                      </label>
                      <input style ={{lineHieght : "25px"}}  
                        class="input--style-4"
                        type="text"
                        name="email"
                        onChange={handleChange}
                      />
                    </div>
                  </div>
                </div>
                <div class="row row-space">
                  <div class="col-12">
                    <div class="input-group">
                      <label class="label" for="phone">
                        Phone number
                      </label>
                      <input style ={{lineHieght : '25px'}}  
                        class="input--style-4"
                        type="text"
                        name="phone"
                        onChange={handleChange}
                      />
                    </div>
                  </div>
                </div>
                <div class="row row-space">
                  <div class="col-6">
                    <div class="input-group">
                      <label class="label" for="password">
                        Password
                      </label>
                      <input style ={{lineHieght : "25px"}}  
                        class="input--style-4"
                        type="password"
                        name="password"
                        onChange={handleChange}
                      />
                    </div>
                  </div>
                  <div class="col-6">
                    <div class="input-group">
                      <label class="label" for="confirmpassword">
                        Confirm Password
                      </label>
                      <input style ={{lineHieght : "25px"}}  
                        class="input--style-4"
                        type="password"
                        name="confirmpassword"
                        onChange={handleChange}
                      />
                    </div>
                  </div>
                </div>
                {errorMessage && (
                  <div> {errorMessage}</div> 
                )}
                <div class="p-t-15">
                  <button class="btn btn--radius-2 btn--blue" type="submit">
                    Sign Up
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>
      </div>
      <script src="./vendor/register/jquery/jquery.min.js"></script>
      <script src="./vendor/register/select2/select2.min.js"></script>
      <script src="./vendor/register/datepicker/moment.min.js"></script>
      <script src="./vendor/register/datepicker/daterangepicker.js"></script>
      <script src="./js/global.js"></script>
    </div>
  );
}
