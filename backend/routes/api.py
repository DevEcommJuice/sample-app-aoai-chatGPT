# backend/routes/api.py
"""
Blueprint principal de la API InviktaChatDemo.

Aquí se agrupan todas las rutas bajo el prefijo /api:
- /api/ping          → test rápido de vida
- /api/info          → metadatos básicos de la app
- /api/version       → número de versión (si lo defines)
"""

from quart import Blueprint, jsonify
import os

# Creamos el blueprint principal de la API
bp = Blueprint("api", __name__)

@bp.get("/api/ping")
async def ping():
    """Prueba de conexión simple."""
    return jsonify({"ping": "pong"}), 200


@bp.get("/api/info")
async def info():
    """Devuelve información básica de la app."""
    return jsonify({
        "app": "InviktaChatDemo",
        "environment": os.getenv("APP_ENV", "development"),
        "message": "API endpoint principal operativo"
    }), 200


@bp.get("/api/version")
async def version():
    """Ejemplo de endpoint para exponer versión."""
    return jsonify({
        "version": os.getenv("APP_VERSION", "1.0.0"),
        "status": "ok"
    }), 200
