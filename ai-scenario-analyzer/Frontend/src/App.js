import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [scenario, setScenario] = useState('');
  const [constraints, setConstraints] = useState(['']);
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleConstraintChange = (index, value) => {
    const newConstraints = [...constraints];
    newConstraints[index] = value;
    setConstraints(newConstraints);
  };

  const addConstraint = () => {
    setConstraints([...constraints, '']);
  };

  const handleSubmit = async () => {
    setLoading(true);
    try {
      const res = await axios.post('http://localhost:3000/api/analyze-scenario', {
        scenario,
        constraints: constraints.filter(c => c.trim() !== '')
      });
      setResponse(res.data);
    } catch (error) {
      alert("Something went wrong: " + error.message);
    }
    setLoading(false);
  };

  return (
    <div className="App">
      <h1>Scenario Analyzer</h1>

      <textarea
        rows="4"
        placeholder="Enter your scenario..."
        value={scenario}
        onChange={(e) => setScenario(e.target.value)}
      />

      <h3>Constraints</h3>
      {constraints.map((constraint, index) => (
        <input
          key={index}
          type="text"
          placeholder={`Constraint ${index + 1}`}
          value={constraint}
          onChange={(e) => handleConstraintChange(index, e.target.value)}
        />
      ))}
      <button onClick={addConstraint}>+ Add Constraint</button>

      <br /><br />
      <button onClick={handleSubmit} disabled={loading}>
        {loading ? 'Analyzing...' : 'Submit'}
      </button>

      {response && (
        <div className="response">
          <h2>Scenario Summary</h2>
          <p>{response.scenarioSummary}</p>

          <h3>Potential Pitfalls</h3>
          <ul>{response.potentialPitfalls.map((p, i) => <li key={i}>{p}</li>)}</ul>

          <h3>Proposed Strategies</h3>
          <ul>{response.proposedStrategies.map((s, i) => <li key={i}>{s}</li>)}</ul>

          <h3>Recommended Resources</h3>
          <ul>{response.recommendedResources.map((r, i) => <li key={i}>{r}</li>)}</ul>

          <h4>Disclaimer</h4>
          <p>{response.disclaimer}</p>
        </div>
      )}
    </div>
  );
}

export default App;