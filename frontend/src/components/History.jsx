import "../styles/History.css";

function History({ history }) {
  if (!history || history.length === 0) {
    return null;
  }

  return (
    <div className="history-card">
      <h2>Recent Predictions</h2>

      {history.map((item, index) => (
        <div
          key={index}
          className={`history-item ${
            item.prediction === "Good" ? "good-item" : "bad-item"
          }`}
        >
          <span>
            {item.prediction === "Good" ? "🟢" : "🔴"}
          </span>

          <div className="history-text">
            <strong>{item.prediction}</strong>

            <p>{item.sentence}</p>

            <small>{item.confidence}% confidence</small>
          </div>
        </div>
      ))}
    </div>
  );
}

export default History;