// Change this to your deployed backend URL in production
// (e.g. "https://your-api.onrender.com")
const API_BASE_URL = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
  ? "http://localhost:8000"
  : "http://localhost:8000"; // TODO: replace with production API URL after deployment
