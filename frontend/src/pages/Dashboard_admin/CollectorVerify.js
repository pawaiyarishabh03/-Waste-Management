import React, { useState } from "react";
import { FcCheckmark } from "react-icons/fc";
import { RxCross1 } from "react-icons/rx";
import axios from "axios";
export default function CollectorResponse({data, email, getData}) {
  console.log(email)
  const [collectorArr, setCollectorArr] = useState([
    { name: "abcd", email: "jgh5j@gmail.com", phone: "9414461366" },
    { name: "wxyz", email: "rkobgjv@gmail.com", phone: "9414454553" },
    { name: "efgh", email: "relbmvm@gmail.com", phone: "8578057555" },
    { name: "ijkl", email: "klrhmbv@gmail.com", phone: "6565667764" },
    { name: "mnop", email: "rklgbmvlm@gmail.com", phone: "7688768785" },
  ]);
  return (
    <div>
      <table
        class="table table-striped"
        style={{ width: "92%", left: "6%", position: "absolute" }}
      >
        <thead>
          <tr>
            <th scope="col">S.No.</th>
            <th scope="col">Collector Name</th>
            <th scope="col">Collector Email</th>
            <th scope="col">Collector Mobile No.</th>
            <th scope="col">Action</th>
          </tr>
        </thead>

        <tbody id="tableBody">
          {data?.map((collector, index) => (
            <tr>
              <td>{index + 1}</td>
              <td>{collector.name}</td>
              <td>{collector.email}</td>
              <td>{collector.number}</td>
              <td>
                <button
                  style={{ marginRight: "20px" }}
                  onClick={async() => {
                    console.log("Collector approved successfully");
                    await axios.post('http://localhost:8000/verifyCollector', {
                      email : email,
                      collectorId : collector._id
                  })
                  .then(res => {
                    console.log(res);
                    getData(email)
                  })
                  .catch(err => {
                    console.log(err);
                  })
                  }}
                >
                  <FcCheckmark fontSize="1.4em" />
                </button>
                <button
                  onClick={async() => {
                    console.log("Collector approved successfully");
                    await axios.post('http://localhost:8000/rejectCollector', {
                      email : email,
                      collectorId : collector._id
                  })
                  .then(res => {
                    console.log(res);
                    getData(email)
                  })
                  .catch(err => {
                    console.log(err);
                  })
                  }}
                >
                  <RxCross1 color="red" fontSize="1.4em" />
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
