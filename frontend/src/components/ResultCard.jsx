import { motion } from "framer-motion";
import ConfidenceBar from "./ConfidenceBar";
import "../styles/ResultCard.css";

function ResultCard({ result }) {
  if (!result) return null;

  const good = result.prediction === "Good";

  return (
    <motion.div
      className={`result-card ${good ? "good" : "bad"}`}
      initial={{ opacity: 0, y: 40 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <h2>{good ? "🟢 GOOD" : "🔴 BAD"}</h2>

      <ConfidenceBar confidence={result.confidence} />

      <p className="confidence">
        Confidence: <strong>{result.confidence}%</strong>
      </p>

      <div className="sentence-box">
        "{result.sentence}"
      </div>
    </motion.div>
  );
}

export default ResultCard;