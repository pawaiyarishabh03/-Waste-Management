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
            <th scope="col">Date</th>
            <th scope="col">Username</th>
            <th scope="col">Waste Type</th>
            <th scope="col">Status</th>
          </tr>
        </thead>

        <tbody id="tableBody">
          <tr>
            <td>1</td>
            <td>2012-09-31</td>
            <td>abcd</td>
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
          </tr>
          <tr>
            <td>2</td>
            <td>2013-09-31</td>
            <td>abcd</td>
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
          </tr>
          <tr>
            <td>3</td>
            <td>2014-09-31</td>
            <td>abcd</td>
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
          </tr>
          <tr>
            <td>4</td>
            <td>2015-09-31</td>
            <td>abcd</td>
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
          </tr>
          <tr>
            <td>5</td>
            <td>2016-09-31</td>
            <td>abcd</td>
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
          </tr>
        </tbody>
      </table>
    </div>
  );
}
