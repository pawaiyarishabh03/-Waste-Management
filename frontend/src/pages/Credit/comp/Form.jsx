import React from 'react'
import { useState } from 'react';
import "./Form.css"
import axios from 'axios';
import { useEffect } from 'react';
import Button from '@mui/material/Button';
import { storage } from '../../../firebase';
import {
    ref,
    uploadBytesResumable, getDownloadURL
} from "firebase/storage";
import { useNavigate } from 'react-router-dom';
import Modal_prop from '../../../Components/Modal_prop';




// const locationfinder = () => {
//     navigator.geolocation.getCurrentPosition(function (position) {
//         console.log("Latitude is :", position.coords.latitude);
//         setlatitude(position.coords.latitude)
//         console.log("Longitude is :", position.coords.longitude);
//         setlongitude(position.coords.longitude)

//     });
// }




export const DonorForm = () => {

    const [tableactive, settable] = React.useState('hide');
    const [donorNo, setdonorNo] = React.useState(0)
    const [User, setUser] = React.useState(null);
    const navigate = useNavigate()

    const [file, setFile] = useState("");
    const [imgurl, setimgurl] = useState('');
    const [longitude, setlongitude] = useState(0)
    const [latitude, setlatitude] = useState(0)
    const [message, setMessage] = useState('');
    const [quantity, setquantity] = useState(0);
    const [type, settype] = useState("Other");
    const [address, setaddress] = useState('');

    const [open, setOpen] = useState(false)



    //   console.log(imgurl)




    // progress
    const [percent, setPercent] = useState(0);

    // Handle file upload event and update state
    function handleChange(event) {
        console.log(event)
        setFile(event.target.files[0]);
    }
    const locationfinder = () => {
        navigator.geolocation.getCurrentPosition(function (position) {
            console.log("Latitude is :", position.coords.latitude);
            setlatitude(position.coords.latitude)
            console.log("Longitude is :", position.coords.longitude);
            setlongitude(position.coords.longitude)

        });
    }
    // const handlesubmit = () => {

    // }


    const handleUpload = () => {

        setOpen(!open)
        console.log(file)
        if (!file) {
            alert("Please upload an image first!");
        }

        const storageRef = ref(storage, `/files/${file.name}`);

        // progress can be paused and resumed. It also exposes progress updates.
        // Receives the storage reference and the file to upload.
        const uploadTask = uploadBytesResumable(storageRef, file);
        console.log(uploadTask)

        uploadTask.on(
            "state_changed",
            (snapshot) => {
                const percent = Math.round(
                    (snapshot.bytesTransferred / snapshot.totalBytes) * 100
                );

                // update progress
                setPercent(percent);
                console.log("uploaded")
            },
            (err) => console.log(err.message),
            () => {
                // download url
                getDownloadURL(uploadTask.snapshot.ref).then((url) => {
                    setimgurl(url)
                    console.log(url);
                });
            }
        );
    }
    useEffect(() => {
        locationfinder()
        const user = JSON.parse(localStorage.getItem('user'))
        if (user) {
            console.log(user)
            setUser(user)
        }
        else
            navigate('/')
    }, [])

    const handlesubmit = async (e) => {
        e.preventDefault()

        console.log(User)

        await axios.post('http://localhost:8000/addrequest', {
            email: User.email,
            latitude,
            longitude,
            photoUrl: imgurl,
            message
        })
            .then((response) => {
                console.log(response.data);

            })
            .catch((err) => {
                console.log(err)
            })

    }



    return (
        <>
            <div className='donorForm'>
                <div className='form-css'>

                    <div className="form-row">
                        <div className="form-group col-md-6">
                            <label for="inputEmail4">Message</label>
                            <input type="text" className="form-control" id="inputEmail4" onChange={(event) => setMessage(event.target.value)} placeholder="Name of organization" />
                        </div>

                    </div>



                    {/* <div className="form-group">
      <label for="inputAddress2">Address 2</label>
      <input type="text" className="form-control" id="inputAddress2" placeholder="Apartment, studio, or floor" />
    </div> */}
                    {/* <label class="my-1 mr-2" for="inlineFormCustomSelectPref">Type</label> <br />
                    <select class="custom-select my-1 mr-sm-2" id="inlineFormCustomSelectPref" onChange={(event) => settype(event.target.value)}>
                        <option selected>Choose...</option>
                        <option value="Individual">Individual</option>
                        <option value="Restaurant">Restaurant</option>
                        <option value="Hostel">Hostel</option>
                        <option value="Food_Caterer">Food Caterer</option>
                        <option value="Hostel">NGO</option>
                        <option value="Other">Other</option>
                    </select> */}


                    <div className="form-row">
                        {/* <div className="form-group col-md-6">
                        <label for="inputCity">City</label>
                        <input type="text" className="form-control" id="inputCity" />
                    </div>
                    <div className="form-group col-md-4">
                        <label for="inputState">State</label>
                        <select id="inputState" className="form-control">

                            <option value="Andhra Pradesh">Andhra Pradesh</option>
                            <option value="Andaman and Nicobar Islands">Andaman and Nicobar Islands</option>
                            <option value="Arunachal Pradesh">Arunachal Pradesh</option>
                            <option value="Assam">Assam</option>
                            <option value="Bihar">Bihar</option>
                            <option value="Chandigarh">Chandigarh</option>
                            <option value="Chhattisgarh">Chhattisgarh</option>
                            <option value="Dadar and Nagar Haveli">Dadar and Nagar Haveli</option>
                            <option value="Daman and Diu">Daman and Diu</option>
                            <option value="Delhi">Delhi</option>
                            <option value="Lakshadweep">Lakshadweep</option>
                            <option value="Puducherry">Puducherry</option>
                            <option value="Goa">Goa</option>
                            <option value="Gujarat">Gujarat</option>
                            <option value="Haryana">Haryana</option>
                            <option value="Himachal Pradesh">Himachal Pradesh</option>
                            <option value="Jammu and Kashmir">Jammu and Kashmir</option>
                            <option value="Jharkhand">Jharkhand</option>
                            <option value="Karnataka">Karnataka</option>
                            <option value="Kerala">Kerala</option>
                            <option value="Madhya Pradesh">Madhya Pradesh</option>
                            <option value="Maharashtra">Maharashtra</option>
                            <option value="Manipur">Manipur</option>
                            <option value="Meghalaya">Meghalaya</option>
                            <option value="Mizoram">Mizoram</option>
                            <option value="Nagaland">Nagaland</option>
                            <option value="Odisha">Odisha</option>
                            <option value="Punjab">Punjab</option>
                            <option value="Rajasthan">Rajasthan</option>
                            <option value="Sikkim">Sikkim</option>
                            <option value="Tamil Nadu">Tamil Nadu</option>
                            <option value="Telangana">Telangana</option>
                            <option value="Tripura">Tripura</option>
                            <option value="Uttar Pradesh">Uttar Pradesh</option>
                            <option value="Uttarakhand">Uttarakhand</option>
                            <option value="West Bengal">West Bengal</option>
                            <option selected>Choose...</option>
                            <option>...</option>
                        </select>
                    </div> */}
                        <div>
                            <label class="my-1 mr-2" for="inlineFormCustomSelectPref">Upload waste image</label> <br />

                            <input type="file" onChange={handleChange} accept="/image/*" />
                            <br />
                            <br />
                            <button onClick={handleUpload}>Upload to Firebase</button>
                            {/* <p>{percent} "% done"</p> */}


                            {/* {imgurl && <>
                                <br />
                                <br />
                                <img className='image-height' src={imgurl} alt="" />

                            </>} */}
                        </div>


                    </div>
                    <br />
                    <button type="submit" className="btn btn-primary" onClick={handlesubmit}>Submit</button>
                </div>

                {open && <Modal_prop imgurl = {imgurl} />}

            </div>
            <div className="listContainer">
                {/* {tableactive=="show" &&
         <>
         <div className="listTitle">Lastest Transactions</div>
            <TableX latitude={latitude} longitude={longitude} donorNo={donorNo} /></>
         } */}
            </div>


        </>



    )
}
