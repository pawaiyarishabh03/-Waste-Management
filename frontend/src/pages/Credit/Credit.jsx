import * as React from 'react';
import { styled, useTheme } from '@mui/material/styles';
import Box from '@mui/material/Box';
import MuiDrawer from '@mui/material/Drawer';
import MuiAppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import DashboardIcon from "@mui/icons-material/Dashboard";
import PersonIcon from "@mui/icons-material/Person";
import LogoutIcon from "@mui/icons-material/Logout";
import List from '@mui/material/List';
import CssBaseline from '@mui/material/CssBaseline';
import Typography from '@mui/material/Typography';
import Divider from '@mui/material/Divider';
import IconButton from '@mui/material/IconButton';
import MenuIcon from '@mui/icons-material/Menu';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import ListItem from '@mui/material/ListItem';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemIcon from '@mui/material/ListItemIcon';
import ListItemText from '@mui/material/ListItemText';
import InboxIcon from '@mui/icons-material/MoveToInbox';
import MailIcon from '@mui/icons-material/Mail';
import { Grid, Paper } from '@mui/material';
import AddShoppingCartOutlinedIcon from '@mui/icons-material/AddShoppingCartOutlined';
import Grid_comp from './comp/Grid_comp';
import Chart from './comp/Chart';
import PieChart from './comp/PieChart';
import Table from './comp/Table_comp';
import Table_comp from './comp/Table_comp';
import { DonorForm } from './comp/Form';
import axios from 'axios';
import { useEffect } from 'react';
import { useState } from 'react';
import { Navigate, useNavigate } from 'react-router-dom';

// import UserDistance from './UserDistance';
const drawerWidth = 240;

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

const Credit = () => {
    const navigate = useNavigate()
    const [proData, setProData] = useState([])

    const getProductData = async () => {
        let info = await axios.get("http://localhost:8000/getAllProducts")
        try {
            setProData(info.data)
        } catch (error) {
            console.log(error)
        }
    }

    const [user, setUser] = useState()
    const [score,setScore] = useState(5)
    const getCredit = async(email) => {
        axios.post('http://localhost:8000/getCredit', {
            email : email,
        })
        .then((response) => {
            console.log(response.data)
            setScore(response.data.credit)
        }
        )
        .catch((error) => {
            console.log(error)
        }
        )
    }

    useEffect(() => {
        const token = JSON.parse(localStorage.getItem("token"));
        const User = JSON.parse(localStorage.getItem("user"));
        if (token && User) {
            setUser(User)
        }
        else
        {
            navigate('/')
        }
        getProductData()
        getCredit(User.email)
    }, [])



    const theme = useTheme();
    const [open, setOpen] = React.useState(false);

    const handleDrawerOpen = () => {
        setOpen(true);
    };

    const handleDrawerClose = () => {
        setOpen(false);
    };

    

    return (
        <>

            <Typography variant="h3" noWrap component="div" style={{ marginTop: "10rem", textAlign: "center" }}>
                Welcome to the recyclable product's shop!
            </Typography>

            <Typography variant="h5" noWrap component="div" style={{ textAlign: "center", marginTop: "2rem" }}>
                Your Current Credit Score is: {score}
            </Typography>

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
                        {/* <ListItemButton
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
                                <InboxIcon />
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
                                <InboxIcon />
                            </ListItemIcon>
                            <ListItemText primary="Profile" sx={{ opacity: open ? 1 : 0 }} />
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
                                <InboxIcon />
                            </ListItemIcon>
                            <ListItemText primary="Logout" sx={{ opacity: open ? 1 : 0 }} />
                        </ListItemButton> */}
                    </List>
                </Drawer>



                <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
                    <DrawerHeader />


                    <Grid container spacing={6}>
                        {proData.map((products) => {
                            return (
                                <>

                                    <Grid_comp
                                        title={products.name}
                                        desc={products.description}
                                        imgurl={products.imgurl}
                                        price={products.price}
                                        quantity={products.quantity} 
                                        score = {score}
                                        email = {user.email}
                                        setScore = {setScore}/>

                                </>
                            )
                        })}

                    </Grid>

                    {/* <Grid container spacing={6}>
                        <Grid_comp />
                        <Grid_comp />
                        <Grid_comp />

                    </Grid>

                    <Grid container spacing={6} style={{ marginTop: "5rem" }}>
                        <Grid_comp />
                        <Grid_comp />
                        <Grid_comp />

                    </Grid>

                    <Grid container spacing={6} style={{ marginTop: "5rem" }}>
                        <Grid_comp />
                        <Grid_comp />
                        <Grid_comp />

                    </Grid> */}


                </Box>
            </Box >
        </>
    )

}

export default Credit;
