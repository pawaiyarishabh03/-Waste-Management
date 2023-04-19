import "./Pie.css";
import MoreVertIcon from "@mui/icons-material/MoreVert";
import { CircularProgressbar } from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";
import KeyboardArrowDownIcon from "@mui/icons-material/KeyboardArrowDown";
import KeyboardArrowUpOutlinedIcon from "@mui/icons-material/KeyboardArrowUpOutlined";

const PieChart = () => {
  return (
    <div className="featured">
      <div className="top">
        <h1 className="title"></h1>
        <MoreVertIcon fontSize="small" />
      </div>
      <div className="bottom">
        <div className="featuredChart">
          <CircularProgressbar value={20} text={"200 Tons"} strokeWidth={5} />
        </div>
        <p className="title">Total Waste collected till date</p>
        <div className="summary">
          <div className="item">
            <div className="itemTitle">Target</div>
            <div className="resultAmount">1000 Tons</div>
          </div>
          <div className="item">
            <div className="itemTitle">Last Week</div>
            <div className="resultAmount">3 Tons</div>
          </div>
          <div className="item">
            <div className="itemTitle">Last Month</div>
            <div className="resultAmount">15 Tons</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PieChart;

