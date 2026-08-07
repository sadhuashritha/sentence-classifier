import { motion } from "framer-motion";
import { FaRobot } from "react-icons/fa";
import "../styles/Header.css";

function Header() {
  return (
    <motion.div
      className="header"
      initial={{ opacity: 0, y: -40 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.7 }}
    >
      <div className="logo">
        <FaRobot />
      </div>

      <h1>AI Sentence Classifier</h1>

      <p>
        Analyze whether a sentence is <b>Good</b> or <b>Bad</b> using Machine
        Learning.
      </p>
    </motion.div>
  );
}

export default Header;