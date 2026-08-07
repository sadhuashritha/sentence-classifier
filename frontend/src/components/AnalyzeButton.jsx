import "../styles/Button.css";
import { FaMagic } from "react-icons/fa";

function AnalyzeButton({ loading, onClick }) {
  return (
    <button
      className={`analyze-btn ${loading ? "loading" : ""}`}
      onClick={onClick}
      disabled={loading}
    >
      {loading ? (
        <>
          <span className="spinner"></span>
          AI is analyzing...
        </>
      ) : (
        <>
          <FaMagic />
          Analyze Sentence
        </>
      )}
    </button>
  );
}

export default AnalyzeButton;