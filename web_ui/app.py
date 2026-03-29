"""Flask Web UI."""
import time
from flask import Flask, render_template, request, jsonify
from shieldrag.data.dataset_loader import Query

def create_app(pipeline, config):
    app=Flask(__name__, template_folder="web_ui/templates", static_folder="web_ui/static")
    @app.route("/")
    def index():
        return render_template("index.html", stats=pipeline.get_stats())
    @app.route("/api/query", methods=["POST"])
    def query():
        data=request.get_json(); text=data.get("query","").strip()
        if not text: return jsonify({"error":"Empty"}),400
        q=Query(f"WEB-{int(time.time()*1000)}", text, tenant_id=data.get("tenant","tenant_A"))
        return jsonify(pipeline.process_query(q))
    @app.route("/api/stats")
    def stats():
        return jsonify(pipeline.get_stats())
    return app
