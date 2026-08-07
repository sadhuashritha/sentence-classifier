import "../styles/ResultCard.css";

function ConfidenceBar({ confidence }) {
  return (
    <div className="progress-container">

      <div
        className="progress-fill"
        style={{
          width: `${confidence}%`
        }}
      ></div>

    </div>
  );
}

export default ConfidenceBar;