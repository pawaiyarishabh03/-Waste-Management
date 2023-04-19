import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getStorage } from "firebase/storage";

const firebaseConfig = {
    apiKey: "AIzaSyA0Lc8l4pPKkeoHFtaounlbgvzf0dt6qgU",
    authDomain: "wastebin-e3c9b.firebaseapp.com",
    projectId: "wastebin-e3c9b",
    storageBucket: "wastebin-e3c9b.appspot.com",
    messagingSenderId: "921716139498",
    appId: "1:921716139498:web:1a782a1b8d5bf8cc2afe93",
    measurementId: "G-PQV6ZJZ1B5"
};

const app = initializeApp(firebaseConfig);


const auth = getAuth(app);
const storage = getStorage(app);

export { auth, storage };

export default app;
