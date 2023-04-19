import React from "react";

export default function CollectorResponse() {
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
            <th scope="col">Request</th>
            <th scope="col">Waste Type</th>
            <th scope="col">Status</th>
            <th scope="col">Distance</th>
          </tr>
        </thead>

        <tbody id="tableBody">
          <tr>
            <td>1</td>
            <td>abcd</td>
            <td>msgx1jjfj4rfhj4hfjf</td>
            <td>Recyclable</td>
            <td>
              <div
                style={{
                  borderStyle: "solid",
                  borderRadius: "15px",
                  backgroundColor: "#e66e6e",
                  maxWidth: "100px",
                  textAlign: "center",
                }}
              >
                Pending
              </div>
            </td>
            <td>76.06</td>
          </tr>
          <tr>
            <td>2</td>
            <td>abcd</td>
            <td>msgoewtghfjueonc</td>
            <td>Recyclable</td>
            <td>
              <div
                style={{
                  borderStyle: "solid",
                  borderRadius: "15px",
                  backgroundColor: "#e66e6e",
                  maxWidth: "100px",
                  textAlign: "center",
                }}
              >
                Pending
              </div>
            </td>
            <td>74.04</td>
          </tr>
          <tr>
            <td>3</td>
            <td>abcd</td>
            <td>lejghfuowewjfoeghfgvfhebgvvrfev</td>
            <td>Non-Recyclable</td>
            <td>
              <div
                style={{
                  borderStyle: "solid",
                  borderRadius: "15px",
                  backgroundColor: "#6bde6b",
                  maxWidth: "100px",
                  textAlign: "center",
                }}
              >
                Accepted
              </div>
            </td>
            <td>80.98</td>
          </tr>
          <tr>
            <td>4</td>
            <td>abcd</td>
            <td>klehbvcitwekjf4oehgfdijrd</td>
            <td>Recyclable</td>
            <td>
              <div
                style={{
                  borderStyle: "solid",
                  borderRadius: "15px",
                  backgroundColor: "#e66e6e",
                  maxWidth: "100px",
                  textAlign: "center",
                }}
              >
                Pending
              </div>
            </td>
            <td>65.05</td>
          </tr>
          <tr>
            <td>5</td>
            <td>abcd</td>
            <td>kleh;bvouhelfcxkjwkhfvcgefw</td>
            <td>Recyclable</td>
            <td>
              <div
                style={{
                  borderStyle: "solid",
                  borderRadius: "15px",
                  backgroundColor: "#e66e6e",
                  maxWidth: "100px",
                  textAlign: "center",
                }}
              >
                Pending
              </div>
            </td>
            <td>45.56</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}
