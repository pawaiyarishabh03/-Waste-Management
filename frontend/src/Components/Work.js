import React from "react";
import PickMeals from "../Assets/recycle.png";
import ChooseMeals from "../Assets/collection.png";
import DeliveryMeals from "../Assets/seggregation.png";

const Work = () => {
  const workInfoData = [
    {
      image: PickMeals,
      title: "Recycle More, Landfill Less",
      text: "Get incentives to help protect the environment with increased recycling and composting efforts.",
    },
    {
      image: ChooseMeals,
      title: "Waste Collection, Elevated",
      text: "Save time and resources with optimized collection thanks to our predictive algorithms.",
    },
    {
      image: DeliveryMeals,
      title: "Segregation Simplified",
      text: "Effortlessly separate waste for better disposal and increased recycling.",
    },
  ];
  return (
    <div className="work-section-wrapper">
      <div className="work-section-top">
        <p className="primary-subheading">What we offer?</p>
        <h1 className="primary-heading" style={{ fontSize: "2em" }}>
          Benefits of Our Waste Management Solution
        </h1>
        <p className="primary-text">
          A Solution That Does More: Upgrade to a smarter, more efficient, and
          sustainable waste management solution.
        </p>
      </div>
      <div className="work-section-bottom">
        {workInfoData.map((data) => (
          <div className="work-section-info" key={data.title}>
            <div className="info-boxes-img-container">
              <img src={data.image} alt="" />
            </div>
            <h2>{data.title}</h2>
            <p>{data.text}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Work;
