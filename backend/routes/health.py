# backend/routes/health.py
"""
Endpoints de liveness/readiness.

- /health  y /api/health   → Liveness: responde rápido sin tocar recursos externos.
- /status  y /api/status   → Readiness: estado ligero de la app (sin dependencias pesadas).

Importante:
- No hagas llamadas a BD, Search ni a APIs en estos handlers para no falsos negativos.
- Mantén respuestas mínimas (JSON pequeño / texto) para que sean baratas y rápidas.
"""

from quart import Blueprint, jsonify

# Blueprint exclusivo para endpoints de salud
# No usamos url_prefix para que /health y /status cuelguen de raíz.
health_bp = Blueprint("health", __name__)

@health_bp.get("/health")
@health_bp.get("/api/health")
async def health():
    """
    Liveness probe.
    Si esto responde 200, el proceso está vivo y atiende peticiones HTTP.
    """
    # Texto o JSON es válido; usamos JSON por uniformidad con el resto de la app.
    return jsonify({"status": "ok"}), 200


@health_bp.get("/status")
@health_bp.get("/api/status")
async def status():
    """
    Readiness probe ligera.
    Útil para saber si la app está lista a nivel básico (sin comprobar dependencias).
    Si quieres, puedes ampliar con flags internos (p.ej., cosmos_ready) sin bloquear.
    """
    # Mantén este payload pequeño; evita incluir información sensible.
    return jsonify({
        "app": "InviktaChatDemo",
        "scope": "api",
    }), 200
