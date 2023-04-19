import "./Grid_comp.css";
import KeyboardArrowUpIcon from "@mui/icons-material/KeyboardArrowUp";
import PersonOutlinedIcon from "@mui/icons-material/PersonOutlined";
import AccountBalanceWalletOutlinedIcon from "@mui/icons-material/AccountBalanceWalletOutlined";
import ShoppingCartOutlinedIcon from "@mui/icons-material/ShoppingCartOutlined";
import MonetizationOnOutlinedIcon from "@mui/icons-material/MonetizationOnOutlined";
import { Button } from "@mui/material";
import { useNavigate } from "react-router-dom";

const Grid_comp = (props) => {
    let data;

    //temporary
    const amount = 100;
    const diff = 20;
    const navigate = useNavigate();



    return (
        <div className="widget" style={{height: "18rem"}}>
            <div className="left">
                <span className="title" style={{ fontSize: "1.4rem" }}>{props.header}</span>
                <span className="title" style={{ fontSize: "0.8rem", position: "relative", bottom: "1.5rem" }}>You have earned  {props.credits} this month , You can redeem your credit by clicking the below button</span>


                <Button loading variant="solid" style={{ backgroundColor: "rgba(21, 205, 116, 0.2)", width: "15rem" , position: "relative" , bottom: "2.8rem" }}
                    onClick= {() => {
                        navigate('/credit')
                    }
                }>
                    {props.button}
                </Button>
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
