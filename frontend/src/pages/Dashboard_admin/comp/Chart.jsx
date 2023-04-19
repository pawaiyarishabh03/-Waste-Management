import "./Chart.css";
import {
  AreaChart,
  Area,
  XAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  YAxis,
  Legend,
  Bar
} from "recharts";

const data = [
  {
    "name": "January",
    "waste-collected": 40,
    "pv": 2400
  },
  {
    "name": "February",
    "waste-collected": 30,
    "pv": 1398
  },
  {
    "name": "March",
    "waste-collected": 20,
    "pv": 9800
  },
  {
    "name": "April",
    "waste-collected": 27,
    "pv": 3908
  },
  {
    "name": "May",
    "waste-collected": 18,
    "pv": 4800
  },
  {
    "name": "June",
    "waste-collected": 23,
    "pv": 3800
  },
  {
    "months": "July",
    "waste-collected": 34,
    "pv": 4300
  }
]

const Chart = ({ aspect, title }) => {
  return (
    <div className="chart">
      <div className="title">{title}</div>
      <ResponsiveContainer width="100%" aspect={aspect} style={{marginBottom: "2rem"}}>
        <BarChart width={730} height={250} data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Bar dataKey="waste-collected" fill="#82ca9d" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default Chart;
