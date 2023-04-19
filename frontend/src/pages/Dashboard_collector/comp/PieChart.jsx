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
        <MoreVertIcon fontSize="small" />
      </div>
      <div className="bottom">
        <div className="featuredChart">
          <CircularProgressbar value={25} text={"2 Tons"} strokeWidth={5} />
        </div>
        <p className="title">Total waste collected today</p>
        <div className="summary">
          <div className="item">
            <div className="itemTitle">Target</div>
            <div className="resultAmount">50 Tons</div>
          </div>
          <div className="item">
            <div className="itemTitle">Last Week</div>
            <div className="resultAmount">12 Tons</div>
          </div>
          <div className="item">
            <div className="itemTitle">Last Month</div>
            <div className="resultAmount">55 Tons</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PieChart;

