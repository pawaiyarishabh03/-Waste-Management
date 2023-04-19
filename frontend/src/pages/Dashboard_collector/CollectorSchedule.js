import React from "react";

export default function CollectorSchedule() {
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
          <tr>
            <td>1</td>
            <td>abcd</td>
            <td>2012-09-31</td>
          </tr>
          <tr>
            <td>2</td>
            <td>abcd</td>
            <td>2013-09-31</td>
          </tr>
          <tr>
            <td>3</td>
            <td>abcd</td>
            <td>2014-09-31</td>
          </tr>
          <tr>
            <td>4</td>
            <td>abcd</td>
            <td>2015-09-31</td>
          </tr>
          <tr>
            <td>5</td>
            <td>abcd</td>
            <td>2016-09-31</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}
