import "./Grid_comp.css";
import KeyboardArrowUpIcon from "@mui/icons-material/KeyboardArrowUp";
import PersonOutlinedIcon from "@mui/icons-material/PersonOutlined";
import AccountBalanceWalletOutlinedIcon from "@mui/icons-material/AccountBalanceWalletOutlined";
import ShoppingCartOutlinedIcon from "@mui/icons-material/ShoppingCartOutlined";
import MonetizationOnOutlinedIcon from "@mui/icons-material/MonetizationOnOutlined";

const Grid_comp = ({count}) => {
    let data;

    //temporary
    const amount = 100;
    const diff = 20;

    

    return (
        <div className="widget" style={{height: "10rem"}}>
            <div className="left">
                <span className="title" style={{fontSize: "1.2rem"}}>Welcome,admin!</span>
                <span className="link">Total number of collectors: {count} </span>
            </div>
            <div className="right">
                <AccountBalanceWalletOutlinedIcon
                    className="icon"
                    style={{
                        backgroundColor: "rgba(128, 0, 128, 0.2)",
                        color: "purple",
                    }}
                />
            </div>
        </div>
    );
};

export default Grid_comp;
