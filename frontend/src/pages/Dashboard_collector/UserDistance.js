import React, { useState } from "react";
import axios from "axios";
import socket from "../../socket";

export default function CollectorResponse(props) {
  // let data = [
  //   {
  //     name: "abcd",
  //     updatedBy: "2000-10-06T22:22:55",
  //     wasteType: "Recyclable",
  //     message: "jregkjfchx4kjgn",
  //     distance: 65.48,
  //   },
  //   {
  //     name: "efgh",
  //     updatedBy: "2022-10-06T22:22:55",
  //     wasteType: "Non-Recyclable",
  //     message: "kjrbjbkrjklbjbyrkg",
  //     distance: 57.48,
  //   },
  //   {
  //     name: "ijkl",
  //     updatedBy: "2021-10-06T22:22:55",
  //     wasteType: "Organic",
  //     message: "yjorhiotrjrhbijt",
  //     distance: 48.45,
  //   },
  //   {
  //     name: "mnop",
  //     updatedBy: "2023-10-06T22:22:55",
  //     wasteType: "Non-Recyclable",
  //     message: "jklnjtjlgnjltnbjlnjn",
  //     distance: 55.47,
  //   },
  //   {
  //     name: "qrst",
  //     updatedBy: "2020-10-06T22:22:55",
  //     wasteType: "Recyclable",
  //     message: "lnbklvnjrlkgnbvllfhngcblfgnhjlgbjdjbhvjdfhvjdfvjfjvjdfvhjfv",
  //     distance: 66.48,
  //   },
  // ];
  const handleChange = (e) => {
    setState((state) => ({
      ...state,
      ...{ [e.target.id]: e.target.value },
    }));
    // console.log(state);
  };
  const requests = props.data;
  const [rerender, setRerender] = useState(false);
  const [state, setState] = useState([]);
  return (
    <div class="container-fluid row" style={{ marginLeft: "60px" }}>
      {requests.map((user, index) => (
        <div
          class="noteCard my-2 mx-2 card"
          style={{ width: "20rem", backgroundColor: "white" }}
        >
          <div class="card-body" style={{ width: "100%" }}>
            {/* <h5 class="card-title" id="title ${index}"><b>${element.noteTitle}</b></h5> */}
            <p class="card-text">Username:{user.user}</p>
            <p class="card-text">Request Date:{user.updatedAt}</p>
            <p class="card-text">Waste Type:{user.wasteType}</p>
            <p class="card-text">Message:{user.message}</p>
            <p class="card-text">Distance:{user.distance}</p>
            <hr />
            <form>
              <label htmlFor={index}>Enter Date and time:</label>
              <input
                type="datetime-local"
                name="time"
                min={user.date}
                id={index + 1}
                onChange={handleChange}
              />
              <button
                onClick={async (e) => {
                  let Datetime = "";
                  e.preventDefault();
                  for (const [key, value] of Object.entries(state)) {
                    if (key == index + 1) {
                      Datetime = value;
                      break;
                    }
                  }
                  if (Datetime != "") {
                    console.log("Request approved by collector");
                    // console.log(Datetime);
                    // requests.filter((item) => item.user != user.user);
                    await axios
                      .post("http://localhost:8000/acceptRequest", {
                        email: props.email,
                        approveTime: Datetime,
                        requestId: user._id,
                      })
                      .then((res) => {
                        console.log(res);
                        socket.emit("requestAccepted", {
                          Datetime: Datetime,
                          user: user.user,
                          email: props.email,
                        });
                        props.getRequests();
                      })
                      .catch((err) => {
                        console.log(err);
                      });
                    setRerender(true);
                  } else {
                    alert("Invalid date or time");
                  }
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
