"""Flask Web UI."""
import os
import time
from flask import Flask, render_template, request, jsonify
from shieldrag.data.dataset_loader import Query

_MAX_QUERY_LEN = 2000  # characters

def create_app(pipeline, config):
    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    app=Flask(__name__, template_folder=template_dir)
    @app.route("/")
    def index():
        return render_template("index.html", stats=pipeline.get_stats())
    @app.route("/health")
    def health():
        return jsonify({"status": "ok", "ready": pipeline.is_ready})
    @app.route("/api/query", methods=["POST"])
    def query():
        data=request.get_json(silent=True) or {}
        text=data.get("query","").strip()
        if not text: return jsonify({"error":"Empty query"}),400
        if len(text) > _MAX_QUERY_LEN:
            return jsonify({"error": f"Query too long (max {_MAX_QUERY_LEN} chars)"}), 400
        q=Query(f"WEB-{int(time.time()*1000)}", text, tenant_id=data.get("tenant","tenant_A"))
        return jsonify(pipeline.process_query(q))
    @app.route("/api/stats")
    def stats():
        return jsonify(pipeline.get_stats())
    return app
