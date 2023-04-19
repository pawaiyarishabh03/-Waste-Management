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




// const locationfinder = () => {
//     navigator.geolocation.getCurrentPosition(function (position) {
//         console.log("Latitude is :", position.coords.latitude);
//         setlatitude(position.coords.latitude)
//         console.log("Longitude is :", position.coords.longitude);
//         setlongitude(position.coords.longitude)

//     });
// }




export const DonorForm = () => {
    const [User, setUser] = React.useState(null);
    const navigate = useNavigate()

    const [file, setFile] = useState("");
    const [imgurl, setimgurl] = useState('');
    const [title, setTitle] = useState('');
    const [desc, setDesc] = useState('');
    const [quantity, setQuantity] = useState(0)
    const [price, setPrice] = useState(0)


    // Handle file upload event and update state
    function handleChange(event) {
        console.log(event)
        setFile(event.target.files[0]);
    }



    const handleUpload = () => {
        console.log(file)
        if (!file) {
            alert("Please upload an image first!");
        }

        const storageRef = ref(storage, `/files/product/${file.name}`);

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

        await axios.post('http://localhost:8000/addcredit', {
            name: title,
            price: price,
            quantity: quantity,
            description: desc,
            imgurl: imgurl,
            email: User.email
        }).then((response) => {
            console.log(response.data);

        })
            .catch((err) => {
                console.log(err)
            })




    }



    return (
        <>
            <div className='donorForm' style = {{
                width : '100%',
                display : 'flex',
                flexDirection : 'column',
                justifyContent : 'center',
                alignItems : 'center',
            }}>
                <div className='form-css'>

                    <div className="form-row" style = {{
                        display : 'flex',
                        flexDirection : 'column',
                        justifyContent : 'center',
                        alignItems : 'center',
                    }}>
                        <div className="form-group col-md-6">
                            <input type="text" className="form-control" id="inputEmail4" onChange={(event) => setTitle(event.target.value)} placeholder="Title of the product" />
                        </div>
                    </div>



                    <div className="form-row" style = {{
                        display : 'flex',
                        flexDirection : 'column',
                        justifyContent : 'center',
                        alignItems : 'center',
                    }}>
                        <div className="form-group col-md-6">
                            <input type="text" className="form-control" id="inputEmail4" onChange={(event) => setDesc(event.target.value)} placeholder="Description of the Product" />
                        </div>
                    </div>

                    <div className="form-row" style = {{
                        display : 'flex',
                        flexDirection : 'column',
                        justifyContent : 'center',
                        alignItems : 'center',
                    }}>
                        <div className="form-group col-md-6" style = {{
                        display : 'flex',
                        flexDirection : 'column',
                        justifyContent : 'center',
                        alignItems : 'center',
                    }}>
                            <input type="text" className="form-control" id="inputEmail4" onChange={(event) => setQuantity(event.target.value)} placeholder="Quantity of the product" />
                        </div>
                    </div>

                    <div className="form-row" style = {{
                        display : 'flex',
                        flexDirection : 'column',
                        justifyContent : 'center',
                        alignItems : 'center',
                    }}>
                        <div className="form-group col-md-6">
                            <input type="text" className="form-control" id="inputEmail4" onChange={(event) => setPrice(event.target.value)} placeholder="Price of the product" />
                        </div>
                    </div>

                    <div className="form-row" style = {{
                        display : 'flex',
                        flexDirection : 'column',
                        justifyContent : 'center',
                        alignItems : 'center',
                        width: '100%'
                    }}>

                        <div>
                            <label class="my-1 mr-2" for="inlineFormCustomSelectPref">Upload Product Image</label> <br />

                            <input type="file" onChange={handleChange} accept="/image/*" />
                            <br />
                            <br />
                            <button onClick={handleUpload} style = {{
                                // border : '2px solid black',
                                padding : '5px 10px',
                                color: 'white',
                                borderRadius : '5px',
                                backgroundColor : '#0069D9',
                            }}>Upload</button>
                        </div>


                    </div>
                    <br />
                    <div style = {{
                        display : 'flex',
                        flexDirection : 'column',
                        justifyContent : 'center',
                        alignItems : 'center',
                        width: '100%'
                    }}>
                        
                    <button type="submit" className="btn btn-primary" onClick={handlesubmit}>Submit</button>
                    </div>
                </div>

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
