import React, { useState } from "react";

export default function CollectorSchedule() {
  const [collectorsSchedule, setCollectorsSchedule] = useState([
    { name: "abcd", date: "2023-03-29" },
    { name: "wxyz", date: "2022-03-29" },
    { name: "efgh", date: "2021-03-29" },
    { name: "ijkl", date: "2020-03-29" },
    { name: "mnop", date: "2019-03-29" },
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
            <th scope="col">Username</th>
            <th scope="col">Date</th>
          </tr>
        </thead>

        <tbody id="tableBody">
          {collectorsSchedule.map((schedule, index) => (
            <tr>
              <td>{index + 1}</td>
              <td>{schedule.name}</td>
              <td>{schedule.date}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
