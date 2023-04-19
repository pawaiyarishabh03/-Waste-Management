import React from 'react'
import { MapContainer } from 'https://cdn.esm.sh/react-leaflet/MapContainer'
import { TileLayer } from 'https://cdn.esm.sh/react-leaflet/TileLayer'
import { MarkerLayer, Marker } from "react-leaflet-marker";


const position = [51.505, -0.09]

const Api = () => {
    return (
        <div>
            <MapContainer center={position} zoom={13} scrollWheelZoom={false}>
                <TileLayer
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />
                <Marker position={position}>
                    
                </Marker>
            </MapContainer>

        </div>
    )
}

export default Api
