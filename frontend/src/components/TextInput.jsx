import "../styles/TextInput.css";

const examples = [
  "Thank you for your support.",
  "I hate you.",
  "Have a wonderful day.",
  "I will kill you."
];

function TextInput({ text, setText }) {
  return (
    <div className="text-section">

      <textarea
        className="text-input"
        placeholder="Type or paste your sentence here..."
        value={text}
        maxLength={500}
        onChange={(e) => setText(e.target.value)}
      />

      <div className="text-toolbar">

        <span className="counter">
          {text.length} / 500
        </span>

        <button
          className="clear-btn"
          onClick={() => setText("")}
        >
          Clear
        </button>

      </div>

      <div className="examples">

        <p>Try Examples</p>

        {examples.map((item, index) => (

          <button
            key={index}
            className="example-chip"
            onClick={() => setText(item)}
          >
            {item}
          </button>

        ))}

      </div>

    </div>
  );
}

export default TextInput;