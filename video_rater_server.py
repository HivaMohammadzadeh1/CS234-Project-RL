#!/usr/bin/env python3
"""
Simple Flask server for the video rating app.

Usage:
    python video_rater_server.py

Then open http://localhost:5000 in your browser.
"""

import json
import os
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configuration
VIDEO_DIR = Path("./sample_videos")
RANKINGS_FILE = Path("./video_rankings.json")
PROMPTS_FILE = Path("./video_generation/prompts_8x.json")


@app.route('/')
def index():
    return send_file('video_rater.html')


@app.route('/api/videos')
def get_videos():
    """Return list of all video files."""
    videos = sorted([f.name for f in VIDEO_DIR.glob("*.mp4")])
    return jsonify({"videos": videos})


@app.route('/api/prompts')
def get_prompts():
    """Return list of prompts."""
    try:
        with open(PROMPTS_FILE) as f:
            prompts = json.load(f)
        return jsonify(prompts)
    except FileNotFoundError:
        return jsonify([])


@app.route('/videos/<path:filename>')
def serve_video(filename):
    """Serve video files."""
    return send_from_directory(VIDEO_DIR, filename)


@app.route('/api/save_rankings', methods=['POST'])
def save_rankings():
    """Save rankings to file."""
    data = request.json

    # Load existing rankings
    if RANKINGS_FILE.exists():
        with open(RANKINGS_FILE) as f:
            all_rankings = json.load(f)
    else:
        all_rankings = {}

    # Add timestamp
    data['timestamp'] = datetime.now().isoformat()

    # Save under group key
    group_key = f"group_{data['group']}"
    all_rankings[group_key] = data

    # Write back to file
    with open(RANKINGS_FILE, 'w') as f:
        json.dump(all_rankings, f, indent=2)

    print(f"Saved rankings for group {data['group']}")
    return jsonify({"status": "success"})


@app.route('/api/get_rankings')
def get_rankings():
    """Get all saved rankings."""
    if RANKINGS_FILE.exists():
        with open(RANKINGS_FILE) as f:
            return jsonify(json.load(f))
    return jsonify({})


if __name__ == '__main__':
    print("=" * 60)
    print("Video Ranking Tool Server")
    print("=" * 60)
    print(f"Video directory: {VIDEO_DIR.absolute()}")
    print(f"Rankings will be saved to: {RANKINGS_FILE.absolute()}")
    print(f"\nOpen http://localhost:5001 in your browser")
    print("=" * 60)

    app.run(host='0.0.0.0', port=5001, debug=True)
