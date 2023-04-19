import "./Grid_comp.css";
import KeyboardArrowUpIcon from "@mui/icons-material/KeyboardArrowUp";
import PersonOutlinedIcon from "@mui/icons-material/PersonOutlined";
import AccountBalanceWalletOutlinedIcon from "@mui/icons-material/AccountBalanceWalletOutlined";
import ShoppingCartOutlinedIcon from "@mui/icons-material/ShoppingCartOutlined";
import MonetizationOnOutlinedIcon from "@mui/icons-material/MonetizationOnOutlined";
import { Button } from "@mui/material";
import waste_about from "../../../Assets/waste_about.jpg"
import { useEffect, useState } from "react";
import { async } from "@firebase/util";
import axios from "axios";

const Grid_comp = (props) => {



    const [realprice, setRealPrice] = useState(props.price)
    const [realquantity, setRealQuantity] = useState(props.quantity)
    const [realscore, setRealScore] = useState(props.score)

    const handleQuantity = () => {
        // setRealQuantity(realquantity-1)
    }

    const handlePrice = () => {

        //  if(realscore >= realprice){
        //     setRealScore(realscore-realprice)
        //     props.setScore(realscore-realprice)
        //     alert("purchase successful!")
        //  }

        //  else{
        //     alert("You do not have enough Credits to purchase the requested Product! Wanna purchase by spending real money? click here!")
        //  }
    }

    const changeScore = async(score) => {
        await axios.post("http://localhost:8000/setScore", {
            score: score,
            email : props.email
        })
        .then((res) => {
            console.log(res)
            localStorage.setItem("user", JSON.stringify(res.data))
        }
        )
        .catch((err) => {
            console.log(err)
        }
        )
    }

    const handleBuy = () => {
        setRealQuantity(realquantity - 1)

        if (realscore >= realprice) {
            setRealScore(realscore - realprice)
            changeScore(realscore - realprice)
            props.setScore(realscore - realprice)
            alert("purchase successful!")
        }

        else {
            alert("You do not have enough Credits to purchase the requested Product! Wanna purchase by spending real money? click here!")
        }
    }




    return (
        <div className="widget" style={{ height: "30rem" }}>
            <div className="left">

                <span className="title" style={{ textAlign: "center", marginLeft: "3rem", fontSize: "1rem" }}>{props.title}</span>
                <span className="title" style={{ textAlign: "center", marginLeft: "3rem", fontSize: "0.7rem", position: "relative", bottom: "8.5rem" }}>{props.desc}</span>



                <div className="image" style={{ width: "250px", marginLeft: "14.8rem", marginTop: "-17rem", width: "200px", height: "100px", alignItems: "center" }}>
                    <img src={props.imgurl} alt="" style={{ height: "200px", width: "280px", borderRadius: "15px" }} />
                </div>

                <div className="buttons" style={{ display: "flex", justifyContent: "space-between", width: "100%" }}>

                    <Button onClick={handleBuy} loading variant="solid" style={{ backgroundColor: "rgba(21, 205, 116, 0.2)", width: "9rem", height: "2.3rem" , marginLeft: "2rem" , borderRadius: "10px" }}>
                        Buy Now!
                    </Button>

                    <Button onClick={handleQuantity} loading variant="solid" style={{ backgroundColor: "rgba(173, 166, 19, 0.2)", width: "9rem", marginLeft: "5rem" , height: "2.3rem" , borderRadius: "10px" }}>
                        Quantity: {realquantity}
                    </Button>

                    <Button onClick={handlePrice} loading variant="solid" style={{ backgroundColor: "rgba(128, 0, 128, 0.2)", width: "9rem", marginLeft: "5rem" , height: "2.3rem" , borderRadius: "10px" }}>
                        price: {realprice}
                    </Button>

                </div>


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
        </div >
    );
};

export default Grid_comp;
