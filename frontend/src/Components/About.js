import React from "react";
import AboutBackground from "../Assets/about-background.png";
import AboutBackgroundImage from "../Assets/about-background-image.png";
import { BsFillPlayCircleFill } from "react-icons/bs";
import Waste_about from "../Assets/waste_about.jpg";

const About = () => {
  return (
    <div className="about-section-container">
      {/* <div className="about-background-image-container">
        <img src={AboutBackground} alt="" />
      </div> */}
      <div className="about-section-image-container">
        <img src={Waste_about} alt="" />
      </div>
      <div
        className="about-section-text-container"
        style={{ marginLeft: "20px" }}
      >
        <p className="primary-subheading">About</p>
        <h1 className="primary-heading">
          A <span style={{ color: "#0075c4" }}>Smarter Way</span> to<br></br>{" "}
          Manage your Waste
        </h1>
        <p className="primary-text">
          Upgrade to a smarter and more sustainable waste management solution
          with our advanced technology. Real-time tracking and reporting help
          you make data-driven decisions and reduce waste sent to landfills. Say
          goodbye to traditional waste management methods and join the
          revolution today.
        </p>
        <div className="about-buttons-container">
          <button className="secondary-button"> Know More</button>
        </div>
      </div>
    </div>
  );
};

export default About;
