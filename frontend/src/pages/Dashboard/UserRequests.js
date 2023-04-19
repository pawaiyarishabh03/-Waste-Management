import React, { useState, useEffect } from "react";

export default function UserRequests(props) {
  console.log(props.data);
  return (
    <div>
      <table
        class="table table-striped"
        style={{ width: "92%", left: "6%", position: "absolute" }}
      >
        <thead>
          <tr>
            <th scope="col">S.No.</th>
            <th scope="col">Date</th>
            <th scope="col">Waste Type</th>
            <th scope="col">Status</th>
            <th scope="col">Credits</th>
          </tr>
        </thead>

        <tbody id="tableBody">
          {props.data.map((user, index) => (
            <tr>
              <td>{index + 1}</td>
              <td>{`${new Date(user.updatedAt).getDate()}/${
                new Date(user.updatedAt).getMonth() + 1
              }/${new Date(user.updatedAt).getFullYear()}      ${new Date(
                user.updatedAt
              ).getHours()}:${new Date(user.updatedAt).getMinutes()}:${new Date(
                user.updatedAt
              ).getSeconds()}`}</td>
              <td>{user.wasteType}</td>
              <td>
                <div
                  style={{
                    borderStyle: "solid",
                    borderRadius: "15px",
                    backgroundColor:
                      user.status === "pending" ? "#e66e6e" : "#70d156",
                    maxWidth: "100px",
                    textAlign: "center",
                  }}
                >
                  {user.status}
                </div>
              </td>
              <td>
                {user.wasteType == "Organic" || user.wasteType == "Recyclable"
                  ? 1
                  : 0}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
