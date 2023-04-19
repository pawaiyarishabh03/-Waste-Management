import React from "react";
import BannerBackground from "../Assets/bg-1.png";
import BannerImage from "../Assets/img-home.png";
import Navbar from "./Navbar";
import { FiArrowRight } from "react-icons/fi";
import { useNavigate } from "react-router-dom";

const Home = () => {
  const navigate = useNavigate();
  return (
    <div className="home-container">
      <Navbar />
      <div className="home-banner-container">
        <div className="home-bannerImage-container">
          <img src={BannerBackground} alt="" />
        </div>
        <div className="home-text-section">
          <h1 className="primary-heading">
            Revolutionize Your{" "}
            <span style={{ color: "#0075c4" }}>Waste Management System</span>
          </h1>
          <p className="primary-text">
            Do you struggle with managing and disposing of waste effectively and
            efficiently? With our
            state-of-the-art technology and innovative approach, we make waste
            management simple, efficient, and environmentally friendly.
          </p>
          <button className="secondary-button"
            onClick={() => {
              navigate("/register");
            }}
          >
            Register Now <FiArrowRight />{" "}
          </button>
        </div>
        <div className="home-image-section">
          <img src={BannerImage} style={{ width: "500px" }} alt="" />
        </div>
      </div>
    </div>
  );
};

export default Home;
