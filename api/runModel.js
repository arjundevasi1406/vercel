const express = require('express');
const { exec } = require('child_process');
const path = require('path');

const app = express();

// POST route to execute the Python model
app.post('/api/runModel', (req, res) => {
    const scriptPath = path.join(__dirname, '../model.py');

    // Execute the Python script
    exec(python3 `${scriptPath}`, (err, stdout, stderr) => {
        if (err) {
            return res.status(500).json({ error: 'Error executing model', details: stderr });
        }
        res.status(200).json({ result: stdout.trim() });
    });
});

module.exports = app;