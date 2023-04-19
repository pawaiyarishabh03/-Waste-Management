import "./App.css";
import Home from "./Components/Home";
import About from "./Components/About";
import Work from "./Components/Work";
import Testimonial from "./Components/Testimonial";
import Contact from "./Components/Contact";
import Footer from "./Components/Footer";
import { useEffect } from "react";
import axios from "axios"




function App() {
  return (
    <div className="App">
      <Home />
      <About />
      <Work />
      <Contact />
    </div>
  );
}

export default App;
