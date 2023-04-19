import * as React from "react";
import { styled, useTheme } from "@mui/material/styles";
import Box from "@mui/material/Box";
import MuiDrawer from "@mui/material/Drawer";
import MuiAppBar from "@mui/material/AppBar";
import Toolbar from "@mui/material/Toolbar";
import List from "@mui/material/List";
import CssBaseline from "@mui/material/CssBaseline";
import Typography from "@mui/material/Typography";
import Divider from "@mui/material/Divider";
import IconButton from "@mui/material/IconButton";
import MenuIcon from "@mui/icons-material/Menu";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import ListItem from "@mui/material/ListItem";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemIcon from "@mui/material/ListItemIcon";
import ListItemText from "@mui/material/ListItemText";
import InboxIcon from "@mui/icons-material/MoveToInbox";
import MailIcon from "@mui/icons-material/Mail";
import { Grid, Paper } from "@mui/material";
import AddShoppingCartOutlinedIcon from "@mui/icons-material/AddShoppingCartOutlined";
import Grid_comp from "./comp/Grid_comp";
import Chart from "./comp/Chart";
import PieChart from "./comp/PieChart";
import Table from "./comp/Table_comp";
import Table_comp from "./comp/Table_comp";
import { DonorForm } from "./comp/Form";
import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import UserRequests from "./UserRequests";
import HistoryIcon from "@mui/icons-material/History";
import axios from "axios";
import DashboardIcon from "@mui/icons-material/Dashboard";
import PersonIcon from "@mui/icons-material/Person";
import LogoutIcon from "@mui/icons-material/Logout";
import socket from "../../socket";
import NotificationsActiveIcon from "@mui/icons-material/NotificationsActive";
import NotificationsIcon from "@mui/icons-material/Notifications";
import Backdrop from "@mui/material/Backdrop";
// import Box from '@mui/material/Box';
import Modal from "@mui/material/Modal";
import Fade from "@mui/material/Fade";
import Button from "@mui/material/Button";
import NotificationAddIcon from "@mui/icons-material/NotificationAdd";
// import UserDistance from './UserDistance';
const drawerWidth = 240;

const style = {
  position: "absolute",
  top: "50%",
  left: "50%",
  transform: "translate(-50%, -50%)",
  width: 400,
  bgcolor: "background.paper",
  border: "2px solid #000",
  boxShadow: 24,
  p: 4,
};

const openedMixin = (theme) => ({
  width: drawerWidth,
  transition: theme.transitions.create("width", {
    easing: theme.transitions.easing.sharp,
    duration: theme.transitions.duration.enteringScreen,
  }),
  overflowX: "hidden",
});

const closedMixin = (theme) => ({
  transition: theme.transitions.create("width", {
    easing: theme.transitions.easing.sharp,
    duration: theme.transitions.duration.leavingScreen,
  }),
  overflowX: "hidden",
  width: `calc(${theme.spacing(7)} + 1px)`,
  [theme.breakpoints.up("sm")]: {
    width: `calc(${theme.spacing(8)} + 1px)`,
  },
});

const DrawerHeader = styled("div")(({ theme }) => ({
  display: "flex",
  alignItems: "center",
  justifyContent: "flex-end",
  padding: theme.spacing(0, 1),
  // necessary for content to be below app bar
  ...theme.mixins.toolbar,
}));

const AppBar = styled(MuiAppBar, {
  shouldForwardProp: (prop) => prop !== "open",
})(({ theme, open }) => ({
  zIndex: theme.zIndex.drawer + 1,
  transition: theme.transitions.create(["width", "margin"], {
    easing: theme.transitions.easing.sharp,
    duration: theme.transitions.duration.leavingScreen,
  }),
  ...(open && {
    marginLeft: drawerWidth,
    width: `calc(100% - ${drawerWidth}px)`,
    transition: theme.transitions.create(["width", "margin"], {
      easing: theme.transitions.easing.sharp,
      duration: theme.transitions.duration.enteringScreen,
    }),
  }),
}));

const Drawer = styled(MuiDrawer, {
  shouldForwardProp: (prop) => prop !== "open",
})(({ theme, open }) => ({
  width: drawerWidth,
  flexShrink: 0,
  whiteSpace: "nowrap",
  boxSizing: "border-box",
  ...(open && {
    ...openedMixin(theme),
    "& .MuiDrawer-paper": openedMixin(theme),
  }),
  ...(!open && {
    ...closedMixin(theme),
    "& .MuiDrawer-paper": closedMixin(theme),
  }),
}));

const Dashboard = () => {
  const navigate = useNavigate();
  const [requests, setRequests] = React.useState([]);
  const [user, setUser] = React.useState([]);
  const [isNot, setIsNot] = React.useState(false);
  const [note, setNote] = React.useState("");

  socket.emit("join_room", "room1");
  socket.on("requestAccepted", (data) => {
    if(data.email === user.email){
      console.log(data);
      setIsNot(true);
      setNote(data);
    }
  });
  const getRequests = async () => {
    await axios
      .get("http://localhost:8000/getRequests")
      .then((res) => {
        // console.log(res.data);
        setRequests(res.data);
      })
      .catch((err) => {
        console.log(err);
      });
  };

  useEffect(() => {
    const token = JSON.parse(localStorage.getItem("token"));
    const User = JSON.parse(localStorage.getItem("user"));
    if (token && User && User.role === "User") {
      setUser(User);
      getRequests();
    } else navigate("/");
  }, []);

  const data = requests.filter((item) => item.user === user._id);
  console.log(data);
  const theme = useTheme();
  const [open, setOpen] = React.useState(false);

  const handleDrawerOpen = () => {
    setOpen(true);
  };

  const handleDrawerClose = () => {
    setOpen(false);
  };

  const [open1, setOpen1] = React.useState(false);
  const handleOpen = () => {
    setOpen1(true);
    setIsNot(false);
  };

  const handleClose = () => setOpen1(false);

  const User_details = JSON.parse(localStorage.getItem("user"));

  return (
    <Box sx={{ display: "flex" }}>
      <CssBaseline />
      <AppBar position="fixed" open={open}>
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            onClick={handleDrawerOpen}
            edge="start"
            sx={{
              marginRight: 5,
              ...(open && { display: "none" }),
            }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap component="div">
            Waste Bin
          </Typography>
          <IconButton sx={{ float: "right" }} onClick={handleOpen}>
            {isNot ? (
              <NotificationAddIcon sx={{ color: "white" }} />
            ) : (
              <NotificationsIcon sx={{ color: "white" }} />
            )}
          </IconButton>
          <Modal
            aria-labelledby="transition-modal-title"
            aria-describedby="transition-modal-description"
            open={open1}
            onClose={handleClose}
            closeAfterTransition
            BackdropComponent={Backdrop}
            BackdropProps={{
              timeout: 500,
            }}
          >
            <Fade in={open1}>
              <Box sx={style}>
                <Typography
                  id="transition-modal-title"
                  variant="h4"
                  component="h2"
                >
                  Notifications
                </Typography>
                <Typography id="transition-modal-description" sx={{ mt: 2 }}>
                  "Request has been accepted by the collector"
                </Typography>
              </Box>
            </Fade>
          </Modal>
        </Toolbar>
      </AppBar>
      <Drawer variant="permanent" open={open}>
        <DrawerHeader>
          <IconButton onClick={handleDrawerClose}>
            {theme.direction === "rtl" ? (
              <ChevronRightIcon />
            ) : (
              <ChevronLeftIcon />
            )}
          </IconButton>
        </DrawerHeader>
        <Divider />
        <List>
          <ListItemButton
            sx={{
              minHeight: 48,
              justifyContent: open ? "initial" : "center",
              px: 2.5,
            }}
          >
            <ListItemIcon
              sx={{
                minWidth: 0,
                mr: open ? 3 : "auto",
                justifyContent: "center",
              }}
            >
              <DashboardIcon />
            </ListItemIcon>
            <ListItemText primary="Dashboard" sx={{ opacity: open ? 1 : 0 }} />
          </ListItemButton>
          <ListItemButton
            sx={{
              minHeight: 48,
              justifyContent: open ? "initial" : "center",
              px: 2.5,
            }}
          >
            <ListItemIcon
              sx={{
                minWidth: 0,
                mr: open ? 3 : "auto",
                justifyContent: "center",
              }}
            >
              <HistoryIcon />
            </ListItemIcon>
            <ListItemText primary="History" sx={{ opacity: open ? 1 : 0 }} />
          </ListItemButton>
          <ListItemButton
            sx={{
              minHeight: 48,
              justifyContent: open ? "initial" : "center",
              px: 2.5,
            }}
          >
            <ListItemIcon
              sx={{
                minWidth: 0,
                mr: open ? 3 : "auto",
                justifyContent: "center",
              }}
            >
              <PersonIcon />
            </ListItemIcon>
            <ListItemText primary="Profile" sx={{ opacity: open ? 1 : 0 }} />
          </ListItemButton>
          <ListItemButton
            sx={{
              minHeight: 48,
              justifyContent: open ? "initial" : "center",
              px: 2.5,
            }}
            onClick={() => {
              localStorage.removeItem("token");
              localStorage.removeItem("user");
              navigate("/");
              console.log("logout");
            }}
          >
            <ListItemIcon
              sx={{
                minWidth: 0,
                mr: open ? 3 : "auto",
                justifyContent: "center",
              }}
            >
              <LogoutIcon />
            </ListItemIcon>
            <ListItemText primary="Logout" sx={{ opacity: open ? 1 : 0 }} />
          </ListItemButton>
        </List>
      </Drawer>
      <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
        <DrawerHeader />

        <Grid container spacing={4} style={{ width: "20rem" }}>
          <Grid_comp
            header={`Congratulations , ${User_details.name}!`}
            subheader="You have earned this credits this month , You can redeem your credit by clicking the below button"
            button="Redeem Credits"
          />
          {/* <Grid_comp
            header="Congratulations , User!"
            subheader="You have earned this credits this month , You can redeem your credit by clicking the below button"
            button="Redeem Credits"
          />
          <Grid_comp
            header="Congratulations , User!"
            subheader="You have earned this credits this month , You can redeem your credit by clicking the below button"
            button="Redeem Credits"
          /> */}
        </Grid>

        <h2
          style={{
            position: "relative",
            marginTop: "3rem",
            textAlign: "center",
          }}
        >
          <span style={{ color: "#0075c4" }}>Tackling the Trash:</span>{" "}
          Analyzing Our Progress in Waste Reduction
        </h2>

        <div
          style={{
            display: "flex",
            padding: "5px 20px",
            gap: "20px",
            marignTop: "8rem",
          }}
        >
          <PieChart />
          <Chart />
        </div>

        <div
          className="Form"
          style={{
            position: "relative",
            top: "8rem",
            border: "1px solid black",
            padding: "40px",
            width: "100%",
          }}
        >
          <h2
            style={{ position: "relative", top: "-1rem", textAlign: "center" }}
          >
            Request for Waste Collection
          </h2>
          <DonorForm getRequests = {getRequests} />
          <div />
        </div>
        <div style={{ marginTop: "7rem" }}>
          <h2
            style={{
              position: "relative",
              marginTop: "15rem",
              textAlign: "center",
            }}
          >
            Know your Request Status
          </h2>
          <UserRequests data={data} />
        </div>
      </Box>
    </Box>
  );
};

export default Dashboard;
