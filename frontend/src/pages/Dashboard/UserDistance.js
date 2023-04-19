import React, { useState } from "react";

export default function CollectorResponse() {
  const [userDist, setUserDist] = useState([
    {
      name: "abcd",
      date: "2000-10-06T22:22:55",
      waste_type: "Recyclable",
      msg: "jregkjfchx4kjgn",
      distance: 65.48,
    },
    {
      name: "efgh",
      date: "2022-10-06T22:22:55",
      waste_type: "Non-Recyclable",
      msg: "kjrbjbkrjklbjbyrkg",
      distance: 57.48,
    },
    {
      name: "ijkl",
      date: "2021-10-06T22:22:55",
      waste_type: "Organic",
      msg: "yjorhiotrjrhbijt",
      distance: 48.45,
    },
    {
      name: "mnop",
      date: "2023-10-06T22:22:55",
      waste_type: "Non-Recyclable",
      msg: "jklnjtjlgnjltnbjlnjn",
      distance: 55.47,
    },
    {
      name: "qrst",
      date: "2020-10-06T22:22:55",
      waste_type: "Recyclable",
      msg: "lnbklvnjrlkgnbvllfhngcblfgnhjlgbjdjbhvjdfhvjdfvjfjvjdfvhjfv",
      distance: 66.48,
    },
  ]);
  const today = new Date();
  return (
    <div class="container-fluid row" style={{ marginLeft: "60px" }}>
      {userDist.map((user, index) => (
        <div
          class="noteCard my-2 mx-2 card"
          style={{ width: "20rem", backgroundColor: "white" }}
        >
          <div class="card-body">
            {/* <h5 class="card-title" id="title ${index}"><b>${element.noteTitle}</b></h5> */}
            <p class="card-text">Username:{user.name}</p>
            <p class="card-text">Request Date:{user.date}</p>
            <p class="card-text">Waste Type:{user.waste_type}</p>
            <p class="card-text">Message:{user.msg}</p>
            <p class="card-text">Distance:{user.distance}</p>
            <hr />
            <form>
              <label htmlFor={index}>Enter Date and time:</label>
              <input
                type="datetime-local"
                name="time"
                min={user.date}
                value={`${today.getYear()}-${today.getMonth()}-${today.getDate()}T${today.getHours()}:${today.getMinutes()}:${today.getSeconds()}`}
              />
              <button
                onClick={(e) => {
                  // console.log(
                  //   `${today.getFullYear()}-${today.getMonth()}-${today.getDate()}T${today.getHours()}:${today.getMinutes()}:${today.getSeconds()}`
                  // );
                  e.preventDefault();
                  console.log("Request approved by collector");
                  setUserDist(
                    userDist.filter((item) => item.date != user.date)
                  );
                }}
                class="btn btn-primary mx-2 my-3"
              >
                Add to schedule
              </button>
            </form>
          </div>
        </div>
      ))}
    </div>
  );
}


