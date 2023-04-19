import React, { useState } from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import Register from "./pages/auth/Register";
import Login from "./pages/auth/Login";
import Dashboard from "./pages/Dashboard";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Dashboard_collector from "./pages/Dashboard_collector";
import Dashboard_admin from "./pages/Dashboard_admin";
import Credit from "./pages/Credit/Credit";
// import { registerLicense } from '@syncfusion/ej2-base';

// // Registering Syncfusion license key
// registerLicense('ORg4AjUWIQA/Gnt2VVhkQlFacldJXnxIfUx0RWFab1p6dFFMZVVBJAtUQF1hSn5Rd0djWnpfcHNWQmVU');
const root = ReactDOM.createRoot(document.getElementById("root"));


root.render(

  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/collector" element={<Dashboard_collector />} />
        <Route path="/admin" element={<Dashboard_admin />} />
        <Route path="/credit" element={<Credit />} />
      </Routes>
    </BrowserRouter>
    {/* <App /> */}
  </React.StrictMode>
);
