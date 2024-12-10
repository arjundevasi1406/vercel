const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');

const app = express();
const upload = multer({ dest: '/tmp' });

// POST route to handle image uploads
app.post('/api/upload', upload.single('image'), (req, res) => {
    const { name } = req.body; // image1.jfif or image2.jfif
    const targetPath = path.join(__dirname, '../images', name);

    // Replace the existing image with the new one
    fs.rename(req.file.path, targetPath, (err) => {
        if (err) {
            return res.status(500).json({ error: 'Failed to save image' });
        }
        res.status(200).json({ message: 'Image uploaded successfully' });
    });
});

module.exports = app;