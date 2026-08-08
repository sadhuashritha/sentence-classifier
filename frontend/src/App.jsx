import { useState } from "react";
import axios from "axios";

import Header from "./components/Header";
import TextInput from "./components/TextInput";
import AnalyzeButton from "./components/AnalyzeButton";
import ResultCard from "./components/ResultCard";
import History from "./components/History";

import "./styles/App.css";

function App() {

    const [text, setText] = useState("");
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [history, setHistory] = useState([]);

    const analyzeSentence = async () => {

        if (!text.trim()) {
            alert("Please enter a sentence.");
            return;
        }

        setLoading(true);

        // Hide old result while analyzing
        setResult(null);

        try {

            // Start timer
            const start = performance.now();

            const response = await axios.post(
                "https://sentence-classifier-4tkg.onrender.com/predict",
                {
                text: text
            }
            );

            // End timer
            const end = performance.now();

            // Prediction time
            response.data.time = ((end - start) / 1000).toFixed(2);

            // Save result
            setResult(response.data);

            // Save history
            setHistory((prev) => [
                response.data,
                ...prev.slice(0, 4)
            ]);

        }
        catch (error) {

            console.error(error);
            alert("Cannot connect to Backend.");

        }
        finally {

            setLoading(false);

        }

    };

    return (

        <div className="app">

            <div className="glass-card">

                <Header />

                <TextInput
                    text={text}
                    setText={setText}
                />

                <AnalyzeButton
                    loading={loading}
                    onClick={analyzeSentence}
                />

                <ResultCard
                    result={result}
                />

                <History
                    history={history}
                />

            </div>

        </div>

    );

}

export default App;